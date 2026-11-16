# Lichess bot

The vendored runner in `bot/` is lichess-bot at
`baa2b7e78320b99125a91c2b1805bb7a5c4fb6ea` from
https://github.com/lichess-bot-devs/lichess-bot.
Its AGPL-3.0 license is included in `bot/LICENSE`.
The configuration and explicit challenge helper were migrated from the desktop deployment.

Put the Pawnstar medium int8 model at `models/medium.onnx`.
Expected SHA-256 is
`7a3587112b5ce8f3559f74140aa0627d85afe184c71cba24cd426647836a888b`.

Create a gitignored `.env` in the repository root containing
`LICHESS_BOT_TOKEN=your-token`, then `chmod 600 .env`.
Never commit credentials.

From the repository root, run `./lichess/run.sh`.
This builds the image, replaces only the named `pawnstar-bot` container,
and runs it with 2 CPUs and restart policy `unless-stopped`.
The model is mounted read-only. Configuration is persisted in the image.
Rebuild using the same command after configuration changes.

The bot accepts only standard bullet, blitz, rapid and classical challenges.
Concurrency is 64; correspondence check-in is 5 seconds.
Matchmaking is disabled. Outgoing challenges are sent only explicitly.

Run `docker logs --tail 50 pawnstar-bot` to inspect connection status.

Run `docker exec pawnstar-bot challenge maia1 5+3 rated` to send one challenge.
This command does send a real challenge; there are no automatic retries.
Use `casual` instead of `rated` when desired.
