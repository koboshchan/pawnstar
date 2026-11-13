"""Single-GPU or CPU training. Run from repo root with python -m training.train."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.data import Shards
from training.model import CONFIGS, Config, PolicyValue

def objective(outputs, batch, weight):
    logits, value = outputs
    _, labels, targets, valid = batch
    policy = F.cross_entropy(logits.float(), labels)
    value_loss = (F.binary_cross_entropy_with_logits(value.float(), targets,
                    reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    return policy + weight * value_loss, policy, value_loss

def loader(args, split, repeat=False):
    kwargs = dict(num_workers=args.workers, batch_size=None,
                  pin_memory=args.device.startswith("cuda"))
    if args.workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(Shards(args.data, split, args.batch_size, args.seed, repeat),
                      generator=torch.Generator().manual_seed(args.seed), **kwargs)

def autocast(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()

@torch.inference_mode()
def evaluate(model, args):
    model.eval()
    totals = np.zeros(4)
    for i, batch in enumerate(loader(args, 1)):
        if i >= args.eval_batches:
            break
        batch = tuple(x.to(args.device, non_blocking=True) for x in batch)
        with autocast(args):
            outputs = model(batch[0])
            loss, policy, value = objective(outputs, batch, args.value_weight)
        n = len(batch[0])
        totals += [float(policy)*n, float(value)*n,
                   int((outputs[0].argmax(-1) == batch[1]).sum()), n]
    model.train()
    if not totals[3]:
        raise ValueError("No validation positions found")
    return dict(val_policy=totals[0]/totals[3], val_value=totals[1]/totals[3],
                val_top1=totals[2]/totals[3], val_positions=int(totals[3]))

def save_checkpoint(raw, optimizer, args, step, consumed, history):
    out = Path(args.output)
    path = out / f"step-{step:09d}.pt"
    state = dict(config=asdict(raw.config), model=raw.state_dict(),
                 optimizer=optimizer.state_dict(), step=step, consumed=consumed,
                 args=vars(args), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 python_rng=random.getstate())
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    history[:] = [entry for entry in history if entry != path]
    history.append(path)
    # Keep local files until the new upload is confirmed. Only checkpoint files
    # from this run are ever removed, never the dataset or other run prefixes.
    if args.hf_run:
        destination = "hf://buckets/koboshchan/checkpoints/pawnstar/runs/" + args.hf_run
        subprocess.run(["hf", "buckets", "sync", str(out), destination,
                        "--include", "step-*.pt", "--no-delete"], check=True)
    while len(history) > 2:
        old = history[0]
        if args.hf_run:
            subprocess.run(["hf", "buckets", "rm",
                            destination + "/" + old.name, "-y"], check=True)
        old.unlink(missing_ok=True)
        history.pop(0)
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--config", choices=CONFIGS, default="small")
    p.add_argument("--output", default="runs/small")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=.1)
    p.add_argument("--value-weight", type=float, default=.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--hf-run", help="Dedicated new run name; uploads and keeps last two checkpoints")
    p.add_argument("--wandb", help="Optional W&B project; omitted means disabled")
    args = p.parse_args()
    if any(getattr(args, k) <= 0 for k in ("batch_size", "accum", "steps", "eval_every",
                                           "eval_batches", "save_every", "log_every")):
        p.error("Batch/step/interval arguments must be positive")
    if args.workers < 0 or args.warmup < 0:
        p.error("Workers and warmup must be nonnegative")
    if args.hf_run and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.hf_run):
        p.error("--hf-run must be a simple unique run name")
    if args.hf_run and not shutil.which("hf"):
        p.error("hf CLI is required for uploads")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and list(out.glob("step-*.pt")):
        p.error("Output already has checkpoints; use --resume or a new output directory")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if ckpt:
        for key in ("batch_size", "accum", "workers", "seed", "steps", "warmup", "lr",
                    "weight_decay", "value_weight", "grad_clip", "hf_run"):
            if ckpt["args"][key] != getattr(args, key):
                p.error(f"Resume requires unchanged --{key.replace('_', '-')}")
    raw = PolicyValue(Config(**ckpt["config"]) if ckpt else CONFIGS[args.config]).to(args.device)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, consumed = 0, 0
    if ckpt:
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, consumed = ckpt["step"], ckpt["consumed"]
    model = torch.compile(raw) if args.compile else raw
    model.train()
    stream = iter(loader(args, 0, True))
    # Deterministic replay avoids serializing worker/prefetch state. Can take
    # time for late resumes but never silently repeats already-consumed batches.
    for _ in range(consumed):
        next(stream)
    if ckpt:
        torch.set_rng_state(ckpt["torch_rng"])
        random.setstate(ckpt["python_rng"])
        if args.device.startswith("cuda") and ckpt["cuda_rng"]:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    tracker = None
    if args.wandb:
        import wandb
        tracker = wandb.init(project=args.wandb, config=vars(args))
    history = sorted(out.glob("step-*.pt"))
    print(json.dumps(dict(parameters=sum(x.numel() for x in raw.parameters()),
                          config=asdict(raw.config), start=start)), flush=True)
    begin = time.monotonic()
    for step in range(start+1, args.steps+1):
        if step <= args.warmup:
            factor = step / max(1, args.warmup)
        else:
            progress = min(1., (step-args.warmup)/max(1, args.steps-args.warmup))
            factor = .1 + .9*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr*factor
        optimizer.zero_grad(set_to_none=True)
        metrics = np.zeros(4)
        for _ in range(args.accum):
            try:
                batch = next(stream)
            except StopIteration:
                raise ValueError("No training positions found in assigned shards") from None
            consumed += 1
            batch = tuple(x.to(args.device, non_blocking=True) for x in batch)
            with autocast(args):
                outputs = model(batch[0])
                loss, policy, value = objective(outputs, batch, args.value_weight)
            (loss/args.accum).backward()
            metrics += [float(loss.detach()), float(policy.detach()), float(value.detach()),
                        float((outputs[0].argmax(-1) == batch[1]).float().mean())]
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        if step % args.log_every == 0 or step == 1:
            record = dict(step=step, loss=metrics[0]/args.accum,
                          policy=metrics[1]/args.accum, value=metrics[2]/args.accum,
                          top1=metrics[3]/args.accum, lr=args.lr*factor,
                          grad_norm=float(norm), elapsed=time.monotonic()-begin)
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            record = dict(step=step, **evaluate(model, args))
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=step)
        if step % args.save_every == 0 or step == args.steps:
            path = save_checkpoint(raw, optimizer, args, step, consumed, history)
            print(f"checkpoint {path}", flush=True)
    if tracker:
        tracker.finish()

if __name__ == "__main__":
    main()
