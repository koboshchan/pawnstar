pawnstar training

Run commands from the repository root. Data stays on local disk during training;
the existing HF bucket copy can be downloaded once (about 32.2 GB). Existing
preprocessing output at /content/chess-policy-hf-v1 can be reused without a sync.
Never train while slim conversion or a dataset upload is modifying the files.

Colab GPU setup

Select an A100 GPU manually. These commands do not allocate a runtime.

```python
!git clone https://github.com/koboshchan/pawnstar /content/pawnstar
%cd /content/pawnstar
!pip -q install -r training/requirements.txt
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
# Skip this sync if this dataset is already present from preprocessing.
!hf buckets sync hf://buckets/koboshchan/checkpoints/pawnstar/chess-policy-hf-v1 /content/chess-policy-hf-v1 --no-delete
# Optional one-time full integrity check, reads all 32 GB.
!cd /content/chess-policy-hf-v1 && sha256sum -c SHA256SUMS
```

Start an initial small run. Choose a unique --hf-run name for each new experiment.
The example uses a 1,000-step trial, not an entire dataset pass. It uploads only
checkpoints into that run's dedicated prefix, retaining the latest two there
and locally. Omit --hf-run for strictly local training.

```python
!python -m training.train --data /content/chess-policy-hf-v1 --config small --batch-size 256 --accum 1 --workers 4 --steps 1000 --warmup 100 --eval-every 250 --save-every 250 --output /content/pawnstar-runs/small-1k --hf-run small-1k
```

The same command works on an RTX 4070 Super 12 GB with --batch-size 64 --accum 4
as a conservative starting point. Adjust batch size for actual VRAM usage.
GPU compatibility is implemented but neither GPU has been benchmarked by this
change. GPU runs use bf16 autocast, full-precision optimizer state and gradient
clipping; no fp16 scaler is required. CPU smoke tests run in float32.

The small encoder has 11,441,713 parameters (384 width, 6 layers, 6 heads).
Medium has 50,558,769 (640 width, 10 layers, 10 heads).
Large has 156,179,249 (896 width, 16 layers, 14 heads).
Smaller microbatches and gradient accumulation let medium/large fit limited
VRAM; start at 16/8 respectively on 12 GB and reduce if needed. These are
starting points rather than measured memory guarantees.

Architecture and losses

Every position uses the same 69 active tokens as preprocess/core.py. The
encoder is bidirectional with learned position embeddings, pre-LN blocks and
mean pooling. Policy cross-entropy covers all 1968 moves. Training does not
apply a legal mask; preprocessing already verified labels, and inference
rebuilds and applies the exact legal mask before argmax. The value head predicts
mover-relative win probability through BCE-with-logits. Centipawns use
sigmoid(cp/400); positive/negative mate map to 1/0, not signed probability.
Missing scores are excluded from value loss. Default value weight is 0.1.

AdamW uses cosine decay to 10% of initial learning rate after linear warmup.
--compile optionally wraps the model in torch.compile. --wandb PROJECT enables
optional W&B logging after installing wandb; by default it is not imported.
Logs report total loss, policy/value loss, unmasked top-1 accuracy, gradient
norm, LR and validation policy/value/accuracy. Validation uses only split=1,
training only split=0, and split=2 is untouched. Eval defaults to a deterministic
50-batch subset, not the full validation split.

Each worker reads one NPZ shard at a time, shuffles train shards and rows each
epoch, then yields prebatched tensors. Shards are partitioned between workers
before loading, so workers do not repeat one another's examples. No entire
dataset is held in RAM. 128-token source shards are also accepted if their
unused padding is zero. Split assignments are trusted from preprocessing.

Resume

Use the same data bytes, worker count, batch size, accumulation and schedule.
Point --resume at the latest checkpoint, keeping the other flags unchanged.
Checkpoints include model, AdamW, configuration, optimizer step, consumed
microbatch count and RNG state. Worker streams are deterministically replayed
and skipped to the saved cursor, which can cost significant I/O on late resumes.
Resume rejects changes to the batch/worker/schedule settings. Loading uses
weights_only=True, never arbitrary pickled model objects.

```bash
python -m training.train --data /content/chess-policy-hf-v1 --config small \
  --batch-size 256 --accum 1 --workers 4 --steps 1000 --warmup 100 \
  --eval-every 250 --save-every 250 \
  --output /content/pawnstar-runs/small-1k --hf-run small-1k \
  --resume /content/pawnstar-runs/small-1k/step-000000750.pt
```

After a runtime reset, sync that dedicated HF run prefix back to the local
output directory first. Keep the latest checkpoint and resume it. Do not reuse
an HF run name for an unrelated run. Upload failures stop training with the
new local checkpoint preserved; pruning only begins after successful upload.
Remote removal targets exact checkpoint filenames, never recursive paths.

Play and UCI

```bash
python -m training.play runs/small/step-000001000.pt --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
python -m training.play runs/small/step-000001000.pt --device cuda
```

Without --fen, stdin/stdout speak minimal UCI: uci, isready, ucinewgame,
position startpos/fen plus moves, go, quit. This engine chooses one move without
search; go time controls and depth are ignored. Terminal positions return
bestmove 0000. Black positions mirror ranks and swap colors through the actual
preprocessing encoder; chosen moves are mapped back to the original board.
The dataset has no repetition/fifty-move history features.

Tests and measured smoke proof

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest training/tests -q -s
```

Seven tests cover reference-token/label parity over 106 positions including
black castling, en passant and promotions, split/worker isolation, value
targets, CPU overfit, legal inference/UCI, parameter counts, CLI resume,
local retention and upload-failure-safe remote retention. A two-worker
interrupted CLI run resumes to bit-identical final model tensors.

A separate real-data smoke used the first two local HF-parity sample shards,
copied into fresh /tmp slim shards with 32 train and 16 validation positions.
The actual small 11.44M encoder trained 16 CPU steps, batch 16, LR .002, no
warmup: training loss 8.1238 to 2.8983 in about 12 seconds. Validation policy
loss was 9.1739, expected overfitting on this tiny set, not chess strength.
Resuming step 8 through step 16 reproduced every final model tensor exactly.
The tiny 64-wide unit-test encoder reduced loss 7.6922 to 1.9254 in 25 steps.
No Colab runtime, HF upload, full dataset download or long GPU run was started
during implementation.

## Compatible continuation upgrades

Architecture is unchanged. The default value-loss weight is now 0.5. Policy loss
masks illegal logits before cross-entropy. Shards with packed little-endian
`legal` arrays use those masks; slim shards reconstruct canonical white-to-move
boards in DataLoader workers with python-chess. Invalid masks and illegal
targets fail loudly. Use `--no-masked-policy` only for explicit legacy runs.

Validation remains **unmasked** so `val_policy` and `val_top1` compare directly
with medium-v1/v2. Training loss after masking is not comparable to their old
unmasked training loss. Every evaluation also reports `ema_val_*`.
EMA (default decay 0.9999) keeps an independent fp32 parameter copy, saved as
`ema` alongside raw `model`, optimizer and RNG state. Export and Python play
prefer EMA when present; old checkpoints remain readable. EMA costs about one
extra model's fp32 weights in GPU memory and disk.

`--resume` is still strict for reproducibility. For a new objective/run use
`--fork-from OLD.pt --skip-batches N --output NEW --hf-run NEW`: this retains
raw model and AdamW moments but starts a fresh LR schedule and new counters.
The skip is **absolute** in the original deterministic shard stream, must
be aligned to the unchanged worker count, and must be past all source batches.
After an interrupted source run, include the unsaved tail and a conservative
prefetch margin. Source checkpoints and HF prefixes are never rewritten. A fork
immediately saves/uploads step zero, then follows normal keep-two retention.

### Desktop medium-v3 plan

Source medium-v2 step 105000; same 512 microbatch, accumulation 2, four workers,
seed 42 and bf16 CUDA. The stopped log ended at step 106720; skip 426672 batches
globally (including the old initial 213150 and a conservative unsaved-tail
margin), aligned to four workers. Stop at 182550 new optimizer steps, before
any worker recycles first-epoch data. LR 0.000132, warmup 1000, cosine floor 0.1.
New local output is `E:/pawnstar-runs/medium-v3` because C is nearly full; HF
prefix is `pawnstar/runs/medium-v3`. v1/v2 and the C dataset stay untouched.

The actual slim dataset has no stored masks. A four-worker, 512-row loader
benchmark on unseen shards measured about 63054 positions/s with computed masks
(51200 timed positions, startup excluded), versus about 1380 positions/s for
the previous GPU training. Loader-only results are not end-to-end GPU speed.

### Future experiments, not enabled

SwiGLU/RMSNorm, spatial from-to heads, WDL value heads, history tokens and
multi-PV soft targets need architecture or dataset changes and a separately
planned run. Search and value rerankers are deliberately not part of Pawnstar's
pure searchless engine.
