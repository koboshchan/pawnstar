# pawnstar

searchless transformer chess engine.

- `preprocess/` lichess eval export to token shards (python reference + rust fast path)
- `training/` model + training (todo)
- `engine/` uci engine (todo)


For compact training data, pass `--slim` to the Rust preprocessor, or convert
existing shards in place with `python preprocess/slim.py DATA_DIR --workers 8`.
Then run `python preprocess/verify.py DATA_DIR`. See
[preprocessing documentation](preprocess/README.md) for format and safety details.
