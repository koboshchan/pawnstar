"""Finite offline distillation stage. No runtime allocation or deployment."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from training.model import Config, PolicyValue
from training.rl.search import Network
from training.rl.selfplay import selfplay, save_replay, load_replay
from training.rl.evaluate import evaluate


def load_model(path, device="cpu", require_ema=False, supervised=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if supervised and state.get("stage", "").startswith("rl"):
        raise ValueError("Anchor must be the original supervised checkpoint, not RL weights")
    if require_ema and "ema" not in state:
        raise ValueError("Self-play initialization requires a supervised EMA checkpoint")
    model = PolicyValue(Config(**state["config"]))
    model.load_state_dict(state["ema"] if "ema" in state else state["model"])
    return model.to(device).eval()


def batch(records, indices, device):
    rows = [records[int(i)] for i in indices]
    tokens = torch.as_tensor(np.stack([r[0] for r in rows]), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), 1968), dtype=torch.bool, device=device)
    pi = torch.zeros((len(rows), 1968), device=device)
    for i, row in enumerate(rows):
        ids = torch.as_tensor(row[1], dtype=torch.long, device=device)
        mask[i, ids] = True
        pi[i, ids] = torch.as_tensor(row[2], dtype=torch.float32, device=device)
    value = torch.tensor([r[3] for r in rows], device=device)
    valid = torch.tensor([r[4] for r in rows], device=device)
    return tokens, mask, pi, value, valid


def objective(outputs, anchor_logits, data, kl_weight=1., value_weight=.5):
    _, mask, pi, value, valid = data
    logits, predicted = outputs
    logp = logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
    with torch.no_grad():
        anchor_logp = anchor_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
        anchor_p = anchor_logp.exp()
    policy_loss = -(pi*logp).sum(-1).mean()
    kl = (anchor_p*(anchor_logp-logp)).sum(-1).mean()
    value_loss = (F.binary_cross_entropy_with_logits(predicted.float(), value,
                  reduction="none")*valid).sum()/valid.sum().clamp_min(1)
    loss = policy_loss + kl_weight*kl + value_weight*value_loss
    return loss, dict(policy=float(policy_loss.detach()), kl=float(kl.detach()),
                      value=float(value_loss.detach()), loss=float(loss.detach()))


def fingerprint(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--anchor", help="Original supervised EMA; required after first RL stage")
    p.add_argument("--output", required=True)
    p.add_argument("--replay", help="Reuse a validated NPZ rather than generate games")
    p.add_argument("--device", default="cpu")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--parallel", type=int, default=32)
    p.add_argument("--simulations", type=int, default=32)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-weight", type=float, default=1.)
    p.add_argument("--value-weight", type=float, default=.5)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-pairs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if (min(args.steps, args.batch_size, args.eval_every, args.eval_pairs,
            args.games, args.parallel, args.simulations, args.max_plies) < 1
            or args.lr <= 0 or args.kl_weight <= 0 or args.value_weight < 0):
        p.error("Positive bounds, learning rate and KL weight required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error("Output must be empty; never overwrite a previous run")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    previous = load_model(args.checkpoint, args.device, require_ema=True)
    anchor = load_model(args.anchor or args.checkpoint, args.device, require_ema=True, supervised=True)
    anchor.requires_grad_(False)
    previous.requires_grad_(False)
    model = deepcopy(previous).requires_grad_(True)
    manifest = dict(args=vars(args), checkpoint_sha256=fingerprint(args.checkpoint),
                    anchor_sha256=fingerprint(args.anchor or args.checkpoint),
                    torch=str(torch.__version__))
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.replay:
        records, results = load_replay(args.replay), {"reused": True}
    else:
        records, results = selfplay(Network(previous, args.device), args.games,
                                   args.parallel, args.simulations, args.max_plies, args.seed)
    save_replay(out/"replay.npz", records)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    metrics = []
    for step in range(1, args.steps+1):
        data = batch(records, rng.integers(len(records), size=args.batch_size), args.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            anchor_logits, _ = anchor(data[0])
        loss, report = objective(model(data[0]), anchor_logits, data,
                                 args.kl_weight, args.value_weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite RL loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        report["step"] = step
        if step % args.eval_every == 0 or step == args.steps:
            report["eval"] = evaluate(Network(model, args.device), Network(previous, args.device),
                                      args.eval_pairs, args.max_plies, args.seed+1)
            # Compatible with existing pure Player/exporter. EMA is current distilled weights.
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            path = out/f"step-{step:09d}.pt"
            tmp = path.with_suffix(".tmp")
            torch.save(dict(config=asdict(model.config), model=state, ema=state,
                            step=step, stage="rl-distillation", provenance=manifest), tmp)
            tmp.replace(path)
        metrics.append(report)
        print(json.dumps(report), flush=True)
        (out/"report.json").write_text(json.dumps(dict(results=results,
            positions=len(records), value_positions=sum(r[4] for r in records),
            metrics=metrics), indent=2))


if __name__ == "__main__":
    main()
