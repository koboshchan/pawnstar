# pawnstar

searchless transformer chess engine.

- `preprocess/` lichess eval export to token shards (python reference + rust fast path)
- `training/` policy/value encoder, trainer and minimal UCI player
- `engine/` uci engine (todo)


For compact training data, pass `--slim` to the Rust preprocessor, or convert
existing shards in place with `python preprocess/slim.py DATA_DIR --workers 8`.
Then run `python preprocess/verify.py DATA_DIR`. See
[preprocessing documentation](preprocess/README.md) for format and safety details.

See [training documentation](training/README.md) for GPU/Colab commands, checkpoints and searchless play.
