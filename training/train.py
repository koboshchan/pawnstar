"""Single-GPU or CPU training. Run from repo root with python -m training.train."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from copy import deepcopy
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
    _, labels, targets, valid = batch[:4]
    if len(batch) == 5:
        masks = batch[4]
        if masks.shape != logits.shape or masks.dtype != torch.bool:
            raise ValueError("Invalid legal mask shape or dtype")
        if not masks.gather(1, labels[:, None]).all():
            raise ValueError("Policy target missing from legal mask")
        logits = logits.float().masked_fill(~masks, float("-inf"))
    policy = F.cross_entropy(logits.float(), labels)
    value_loss = (F.binary_cross_entropy_with_logits(value.float(), targets,
                    reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    return policy + weight * value_loss, policy, value_loss

def loader(args, split, repeat=False):
    kwargs = dict(num_workers=args.workers, batch_size=None,
                  pin_memory=args.device.startswith("cuda"))
    if args.workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(Shards(args.data, split, args.batch_size, args.seed, repeat,
                      getattr(args, "skip_batches", 0) if split == 0 else 0,
                      getattr(args, "masked_policy", False) if split == 0 else False),
                      generator=torch.Generator().manual_seed(args.seed), **kwargs)

def autocast(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else nullcontext()

@torch.inference_mode()
def evaluate(model, args):
    was_training = model.training
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
    model.train(was_training)
    if not totals[3]:
        raise ValueError("No validation positions found")
    return dict(val_policy=totals[0]/totals[3], val_value=totals[1]/totals[3],
                val_top1=totals[2]/totals[3], val_positions=int(totals[3]))

def save_checkpoint(raw, optimizer, args, step, consumed, history, ema=None):
    out = Path(args.output)
    path = out / f"step-{step:09d}.pt"
    state = dict(config=asdict(raw.config), model=raw.state_dict(),
                 optimizer=optimizer.state_dict(), step=step, consumed=consumed,
                 args=vars(args), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 python_rng=random.getstate())
    if ema is not None:
        state["ema"] = ema.state_dict()
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
    p.add_argument("--value-weight", type=float, default=.5)
    p.add_argument("--masked-policy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-decay", type=float, default=.9999)
    p.add_argument("--fork-from", help="New run retaining model and optimizer, with a fresh schedule and absolute unseen-data skip")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    p.add_argument("--init-from", help="Weights-only initialization, fresh optimizer and schedule")
    p.add_argument("--skip-batches", type=int, default=0, help="Initial deterministic training batches to omit")
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
    if not 0 <= args.ema_decay < 1:
        p.error("--ema-decay must be in [0, 1)")
    if args.skip_batches < 0 or sum(bool(x) for x in (args.resume, args.init_from, args.fork_from)) > 1:
        p.error("Invalid skip count or conflicting initialization modes")
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
                    "weight_decay", "value_weight", "grad_clip", "hf_run", "masked_policy", "ema_decay"):
            expected = ckpt["args"].get(key, False if key == "masked_policy" else .9999)
            if expected != getattr(args, key):
                p.error(f"Resume requires unchanged --{key.replace('_', '-')}")
    initial = torch.load(args.init_from, map_location="cpu", weights_only=True) if args.init_from else None
    fork = torch.load(args.fork_from, map_location="cpu", weights_only=True) if args.fork_from else None
    source = ckpt or initial or fork
    if fork:
        for key in ("batch_size", "workers", "seed"):
            if fork["args"][key] != getattr(args, key):
                p.error(f"Fork requires unchanged {key}")
        workers = max(1, args.workers)
        safe_skip = (math.ceil(fork["args"].get("skip_batches", 0)/workers)
                     + math.ceil(fork["consumed"]/workers))*workers
        if args.skip_batches % workers or args.skip_batches < safe_skip:
            p.error(f"Fork requires worker-aligned --skip-batches >= {safe_skip}")
        if Path(args.output).resolve() == Path(args.fork_from).resolve().parent:
            p.error("Fork requires a new output directory")
        if args.hf_run and args.hf_run == fork["args"].get("hf_run"):
            p.error("Fork requires a new HF prefix")
    if ckpt and ckpt["args"].get("skip_batches", 0) != args.skip_batches:
        p.error("Resume requires unchanged --skip-batches")
    if initial and args.skip_batches:
        for key in ("batch_size", "workers", "seed"):
            if initial["args"][key] != getattr(args, key):
                p.error(f"Skipping source batches requires unchanged {key}")
        if initial["consumed"] != args.skip_batches:
            p.error("Skip count must match source consumed batches")
    raw = PolicyValue(Config(**source["config"]) if source else CONFIGS[args.config]).to(args.device)
    if initial or fork:
        raw.load_state_dict((initial or fork)["model"])
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if fork:
        optimizer.load_state_dict(fork["optimizer"])
    start, consumed = 0, 0
    if ckpt:
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, consumed = ckpt["step"], ckpt["consumed"]
    ema = deepcopy(raw).eval().requires_grad_(False)
    if source and "ema" in source:
        ema.load_state_dict(source["ema"])
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
    if args.init_from or args.fork_from:
        print(json.dumps(dict(step=0, **evaluate(model, args))), flush=True)
    if args.fork_from:
        path = save_checkpoint(raw, optimizer, args, 0, 0, history, ema)
        print(f"checkpoint {path}", flush=True)
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
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), raw.parameters()):
                averaged.lerp_(current, 1-args.ema_decay)
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
            record.update({"ema_" + k: v for k, v in evaluate(ema, args).items()})
            print(json.dumps(record), flush=True)
            if tracker:
                tracker.log(record, step=step)
        if step % args.save_every == 0 or step == args.steps:
            path = save_checkpoint(raw, optimizer, args, step, consumed, history, ema)
            print(f"checkpoint {path}", flush=True)
    if tracker:
        tracker.finish()

if __name__ == "__main__":
    main()
