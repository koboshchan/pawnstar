# Chess policy preprocessing

Run with Python 3 and numpy, python-chess, requests, zstandard. This implementation reads the ORIGINAL Lichess JSONL export, not the flattened Hugging Face parquet. The flattened data lacks an explicit PV index, so it cannot reliably recover pvs[0] by score sorting. Source https://database.lichess.org/lichess_db_eval.jsonl.zst is mutable; download and sha256-pin the export before a reproducible full build.

Select maximum-depth evaluation, with maximum knodes then first occurrence as deterministic ties. Preserve its first principal variation. Invalid orthodox boards, empty PVs and illegal first moves are rejected, not replaced by weaker evaluations.

Source cp and mate are White-relative, documented at https://lichess.org/api/openapi.yaml. Output scores are mover-relative. Standard chess only. python-chess normalizes king-to-rook castling before black rank mirroring and color swapping. Output move vocabulary is 1968 sorted UCI strings including both promotion directions.

Each shard has uint8 tokens [N,128], 69 active positions, uint16 policy labels, 246-byte little-bit-order packed legal masks, canonical-position uint64 hashes, float32 cp with NaN for absent, int16 mate with zero for absent, and uint8 split (0 train,1 validation,2 test). Piece tokens 1 empty,2..7 white P..K,8..13 black P..K; 14/15 castling false/true; 16 no EP,17..24 EP files. Padding token 0. Squares a1..h8. Castling white king/queen then black king/queen. Only black-to-move positions are mirrored.

Canonical-position hashes assign 98/1/1 splits; duplicate positions cannot cross splits, but this streaming version does not globally deduplicate repeated original records. FEN lacks move counters/history so repetition and fifty-move state are unavailable.

Sample command

    .venv/bin/python preprocess.py --input https://database.lichess.org/lichess_db_eval.jsonl.zst --output sample-original-1m --limit 1000000

Full command, NOT executed

    sha256sum lichess_db_eval.jsonl.zst
    .venv/bin/python preprocess.py --input lichess_db_eval.jsonl.zst --output full --revision SHA256_FROM_ABOVE
    .venv/bin/python verify.py full

Tests

    PYTHONPATH=. .venv/bin/pytest -q
