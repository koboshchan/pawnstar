#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
test -s .env || { echo "Create .env with LICHESS_BOT_TOKEN first" >&2; exit 1; }
test -s models/medium.onnx || { echo "Missing models/medium.onnx" >&2; exit 1; }
chmod 600 .env
docker build -t pawnstar-lichess-bot:local .
if docker container inspect pawnstar-bot >/dev/null 2>&1; then
    docker rm -f pawnstar-bot
fi
exec docker run -d --name pawnstar-bot --cpus 2 --restart unless-stopped     --env-file "$ROOT/.env"     --mount "type=bind,src=$ROOT/models/medium.onnx,dst=/model/medium.onnx,readonly"     pawnstar-lichess-bot:local
