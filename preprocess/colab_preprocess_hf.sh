#!/usr/bin/env bash
# CPU High-RAM Colab. Reuses /content/lichess-hf; no runtime allocation or upload.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
export CHESS_HF_REVISION="${CHESS_HF_REVISION:-8c5285259fdbf3218bf2582de9775dd1378f520b}"
export CHESS_HF_INPUT="${CHESS_HF_INPUT:-/content/lichess-hf}"
export CHESS_OUTPUT="${CHESS_OUTPUT:-/content/chess-policy-hf-v1}"
python -m pip install -q 'huggingface_hub[hf_xet]>=1.0' numpy
python prepare_hf.py
# Conservative output reserve: one output row for every unique contiguous FEN.
python - <<'PY'
import os, shutil
if shutil.disk_usage(os.environ["CHESS_HF_INPUT"]).free < 70_000_000_000:
    raise SystemExit("Need 70GB free for full unmasked output (no JSONL expansion needed)")
PY
if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/chess-rustup.sh
  sh /tmp/chess-rustup.sh -y --profile minimal --default-toolchain stable
fi
rustup toolchain install stable --profile minimal
rustup default stable
cargo build --release --locked --manifest-path rust/Cargo.toml
rust/target/release/chess-preprocess \
  --input "$CHESS_HF_INPUT" --input-format hf-parquet --revision "$CHESS_HF_REVISION" \
  --output "$CHESS_OUTPUT" --vocab moves.json --threads 8 --no-masks \
  --batch-size 20000 --shard-size 100000 --limit "${CHESS_ROWS:-0}"
cp "$CHESS_HF_INPUT/pawnstar-source.json" "$CHESS_OUTPUT/source.json"
python verify.py "$CHESS_OUTPUT"
echo "HF preprocessing complete at $CHESS_OUTPUT. No upload started."
