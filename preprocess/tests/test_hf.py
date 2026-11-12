"""HF row order, shard-boundary carry, limits, and byte-identical JSONL parity."""
import json
import subprocess
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_rust import binary, ROOT

REVISION = "8c5285259fdbf3218bf2582de9775dd1378f520b"
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -",
    "4k3/P7/8/8/8/8/8/4K3 w - -",
    "4k3/8/8/8/8/8/p7/4K3 b - -",
]
SCHEMA = pa.schema([
    ("fen", pa.string()), ("line", pa.string()), ("depth", pa.uint8()),
    ("knodes", pa.int32()), ("cp", pa.int16()), ("mate", pa.int8()),
])


def row(fen, line, depth=30, knodes=100, cp=0, mate=None):
    return dict(fen=fen, line=line, depth=depth, knodes=knodes, cp=cp, mate=mate)


def run(binary, source, out, *args, check=True):
    return subprocess.run([
        str(binary), "--input", str(source), "--output", str(out),
        "--vocab", str(ROOT / "moves.json"), "--batch-size", "1", "--shard-size", "2",
        *args,
    ], check=check, capture_output=True, text=True)


@pytest.mark.parametrize("masks", [False, True])
@pytest.mark.parametrize("limit", [0, 1, 3])
def test_hf_parity(binary, tmp_path, masks, limit):
    rows = [
        row(FENS[0], "d2d4", depth=29, knodes=999),
        row(FENS[0], "c2c4", knodes=99),
        row(FENS[0], "e2e4", cp=3),
        row(FENS[0], "g1f3", cp=99),  # tied eval, first PV wins
        row(FENS[1], "e7e5", cp=0),  # black signed-zero parity
        row(FENS[1], "c7c5", cp=100),
        row(FENS[2], "a7a8n", cp=None, mate=3),
        row(FENS[3], "a2a1q", cp=None, mate=-2),
        row(FENS[0], "e2e5", depth=99),  # illegal selected move, no fallback
        row(FENS[0], "e2e4", depth=30),
    ]
    source = tmp_path / "input"
    (source / "data").mkdir(parents=True)
    # Split winning evaluation/PV across boundaries, including one-row files.
    for i, (start, end) in enumerate([(0, 2), (2, 3), (3, 5), (5, 9), (9, 10)]):
        pq.write_table(pa.Table.from_pylist(rows[start:end], schema=SCHEMA),
                       source / "data" / f"data_{i:04d}.parquet", row_group_size=1)
    from itertools import groupby
    original = []
    for fen, group in groupby(rows, key=lambda r: r["fen"]):
        original.append(dict(fen=fen, evals=[
            dict(depth=r["depth"], knodes=r["knodes"],
                 pvs=[dict(line=r["line"], cp=r["cp"], mate=r["mate"])])
            for r in group
        ]))
    jsonl = tmp_path / "input.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in original))
    extra = ["--limit", str(limit)] + (["--masks"] if masks else [])
    run(binary, jsonl, tmp_path / "jsonl", *extra)
    run(binary, source, tmp_path / "hf", "--input-format", "hf-parquet",
        "--revision", REVISION, *extra)
    left = sorted((tmp_path / "jsonl").glob("shard-*.npz"))
    right = sorted((tmp_path / "hf").glob("shard-*.npz"))
    assert len(left) == len(right)
    for a, b in zip(left, right):
        with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
            assert za.namelist() == zb.namelist()
            for name in za.namelist():
                assert za.read(name) == zb.read(name)
    a = json.loads((tmp_path / "jsonl" / "manifest.json").read_text())
    b = json.loads((tmp_path / "hf" / "manifest.json").read_text())
    for name in ["rows", "accepted", "rejected", "errors", "train", "val", "test", "shards"]:
        assert a[name] == b[name]
    assert b["source_revision"] == REVISION
    assert b["input_format"] == "hf-parquet"
    assert len(b["source_files"]) == 5


def test_hf_requires_revision(binary, tmp_path):
    result = run(binary, tmp_path, tmp_path / "out", "--input-format", "hf-parquet", check=False)
    assert result.returncode != 0
    assert "exact 40-character HF commit SHA" in result.stderr


def test_hf_corruption_fails_not_rejects(binary, tmp_path):
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    result = run(binary, bad, tmp_path / "out", "--input-format", "hf-parquet",
                 "--revision", REVISION, check=False)
    assert result.returncode != 0
    assert not (tmp_path / "out" / "manifest.json").exists()
