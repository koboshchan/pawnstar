"""Reconstruct JSONL from the same ordered HF rows for independent input-mode parity."""
import argparse
from itertools import groupby
import json
from pathlib import Path

import pyarrow.parquet as pq
import zstandard

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path, help="fresh .jsonl.zst path")
parser.add_argument("--limit", type=int, default=1_000_000, help="contiguous FEN groups; 0 for all")
args = parser.parse_args()
directory = args.input / "data" if (args.input / "data").is_dir() else args.input
paths = [args.input] if args.input.is_file() else sorted(directory.glob("*.parquet"))
if not paths:
    raise SystemExit("No parquet shards found")


def rows():
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
            yield from batch.to_pylist()


# Each row is a one-PV evaluation. This preserves max(depth,knodes,earliest)
# without assuming scores are sorted or needing an absent explicit PV index.
with args.output.open("xb") as raw:
    with zstandard.ZstdCompressor().stream_writer(raw) as output:
        for i, (fen, group) in enumerate(groupby(rows(), key=lambda r: r["fen"])):
            if args.limit and i >= args.limit:
                break
            record = dict(fen=fen, evals=[
                dict(depth=r["depth"], knodes=r["knodes"],
                     pvs=[dict(line=r["line"], cp=r["cp"], mate=r["mate"])])
                for r in group
            ])
            output.write((json.dumps(record) + "\n").encode())
