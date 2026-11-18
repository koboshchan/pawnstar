# Training-only RL distillation

This optional offline stage initializes from the supervised v3 EMA and teaches the
same 1968-action policy head to imitate training-time PUCT visit distributions.
The shipped website, UCI player, and Lichess bot remain **pure legal argmax**.
Nothing here is imported by serving code. There is no deployment, allocation,
continuous controller, automatic promotion, or unattended retry.

## Run a finite stage

From the repository root, with the existing training requirements installed:

```sh
python -m pytest training/tests training/rl/tests -q
python -m training.rl.train \
  --checkpoint /path/to/supervised-v3-ema.pt \
  --anchor /path/to/supervised-v3-ema.pt \
  --output runs/rl-stage-001 --device cuda \
  --games 128 --parallel 32 --simulations 32 --max-plies 300 \
  --steps 100 --batch-size 64 --lr 1e-6 --kl-weight 1 \
  --eval-every 50 --eval-pairs 16
```

The checkpoint must contain EMA weights. For a subsequent stage, use the previous
RL checkpoint as `--checkpoint`, but keep `--anchor` fixed to the **original
supervised EMA**. RL-produced anchors are rejected. Only load checkpoints from a
trusted source; loading explicitly uses PyTorch's restricted `weights_only=True`.
Output directories must be empty. The command runs exactly one finite stage.
A validated existing `--replay replay.npz` can replace self-play generation.

## Search and targets

PUCT retains the complete python-chess move stack, including repetition and
draw-claim history. Each simulation selects one leaf per independent root and
batches network inference across roots. Rules, castling, en passant, promotion,
and black-side canonicalization reuse python-chess and Pawnstar's existing
encoder/move vocabulary, not another engine or evaluator.

Values are **side-to-move win probabilities**, not signed scores. The v3 value
logit is sigmoid-transformed; each backup complements the probability at every
ply. Checkmate is 0 for the checkmated side; draws are 0.5. Root priors mix 25%
Dirichlet(alpha=0.3) noise. Moves sample visit counts for the first 24 plies, then
choose the most visited move. Only legal root visit probabilities become policy
targets. Search budgets must be positive.

Completed game outcomes provide white/black-relative value targets of 1, 0,
or 0.5. A move-cap game is marked **truncated**: its policy targets remain usable,
but its value loss is masked out. It is never silently relabeled a draw.
Reports include truncation and value-supervised position counts.

The objective is masked visit-target cross entropy + `kl_weight * KL(anchor || current)`
on legal actions + `value_weight * BCE(outcome)`. The anchor is frozen and stays
in eval mode. Positive KL weight is mandatory. AdamW, a conservative learning
rate, gradient clipping, and nonfinite checks provide additional safeguards.
This does not guarantee improvement or prevent all distribution shift.

Sparse, compressed NPZ replay stores tokens, legal IDs, visit distributions,
outcomes, and value-valid bits. Replay loading forbids pickled arrays and
validates legality and probability normalization.

## Evaluation and artifacts

Every `--eval-every` updates and at the final step, the current network plays the
fixed previous checkpoint using **only legal policy argmax on both sides**.
Seeded random legal openings are paired with colors swapped. There is no tree,
value reranking, root noise, or sampling during match play. Each game has an
explicit ply cap. Truncations are reported separately with null pair scores and
are excluded from completed-game score, not counted as draws. Small smoke
matches and heavily truncated samples are not strength evidence.

Outputs are `manifest.json` (source/anchor SHA256, seed and settings),
`replay.npz`, `report.json`, and atomic `step-*.pt` snapshots compatible with the
existing searchless Player/exporter. The output EMA key deliberately holds the
current distilled parameters, not a second moving average. Periodic snapshots
compare to the fixed input checkpoint; this command does not auto-promote a winner
or run multiple stages. Inspect paired results and use an independent adequate
match before choosing another stage or requesting deployment.

Memory use scales with positions in one stage and roots times simulations.
This is a small auditable reference stage, not a large distributed trainer.
Use a GPU for the medium network. A four-simulation smoke checks plumbing only;
use a meaningful search budget for a separately approved training experiment.
