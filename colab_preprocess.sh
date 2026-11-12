#!/usr/bin/env bash
# Colab cell: !bash /content/chess-transformer/colab_preprocess.sh
# Copy this repository to /content/chess-transformer first. CPU High-RAM, no GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/chess-rustup.sh
  sh /tmp/chess-rustup.sh -y --profile minimal --default-toolchain stable
fi
rustup toolchain install stable --profile minimal
rustup default stable
cargo build --release --locked --manifest-path rust/Cargo.toml
python -m pip install -q numpy requests
export CHESS_EXPORT="${CHESS_EXPORT:-/content/lichess_db_eval.jsonl.zst}"
export CHESS_OUTPUT="${CHESS_OUTPUT:-/content/chess-policy-v1}"
export CHESS_ROWS="${CHESS_ROWS:-0}"
export CHESS_SOURCE_URL="${CHESS_SOURCE_URL:-https://database.lichess.org/lichess_db_eval.jsonl.zst}"
python download_export.py
revision="$(cut -d ' ' -f1 "$CHESS_EXPORT.sha256")"
rust/target/release/chess-preprocess \
  --input "$CHESS_EXPORT" --output "$CHESS_OUTPUT" \
  --revision "$revision" --vocab moves.json --threads 8 \
  --batch-size 20000 --shard-size 100000 --no-masks --limit "$CHESS_ROWS"
cp "$CHESS_EXPORT.source.json" "$CHESS_OUTPUT/source.json"
python verify.py "$CHESS_OUTPUT"
echo "Preprocessing complete. No upload was started."
echo "After approval run python upload_shards.py '$CHESS_OUTPUT' 'hf://buckets/koboshchan/BUCKET/chess-policy-v1'"
