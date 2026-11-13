import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from slim import convert, convert_shard
from verify import verify

from test_rust import binary, rec, ROOT
import chess


def dataset(path, width=128, keys=True, legal=True):
    path.mkdir()
    tokens = np.zeros((3, width), dtype=np.uint8)
    tokens[:, :69] = 1
    arrays = dict(tokens=tokens, labels=np.array([0, 1, 2], dtype="<u2"),
                  cp=np.array([np.nan, -0.0, 3], dtype="<f4"),
                  mate=np.array([0, -2, 3], dtype="<i2"),
                  split=np.array([0, 1, 2], dtype="u1"))
    if keys:
        arrays["keys"] = np.array([0, 980, 990], dtype="<u8")
    if legal:
        arrays["legal"] = np.full((3, 246), 255, dtype="u1")
    np.savez(path / "shard-000000.npz", **arrays)
    (path / "manifest.json").write_text(json.dumps(
        dict(rows=3, accepted=3, rejected=0, shards=1, token_length=width)))
    return arrays


@pytest.mark.parametrize("width", [69, 128])
@pytest.mark.parametrize("keys", [False, True])
@pytest.mark.parametrize("legal", [False, True])
def test_verify_formats(tmp_path, width, keys, legal):
    path = tmp_path / "data"
    dataset(path, width, keys, legal)
    assert verify(path)["verified"] == 3


@pytest.mark.parametrize("field", ["tokens", "split", "keys", "legal", "labels"])
def test_verify_rejects_corruption(tmp_path, field):
    path = tmp_path / "data"
    arrays = dataset(path)
    if field == "tokens":
        arrays[field][0, 100] = 1
    elif field == "split":
        arrays[field][0] = 3
    elif field == "keys":
        arrays[field][0] = 990
    elif field == "legal":
        arrays[field][0] = 0
    else:
        arrays[field][0] = 1968
    np.savez(path / "shard-000000.npz", **arrays)
    with pytest.raises(AssertionError):
        verify(path)


def test_convert(tmp_path):
    path = tmp_path / "data"
    arrays = dataset(path)
    # Exercise mixed original/slim input as after an interrupted conversion.
    convert_shard(path / "shard-000000.npz")
    np.savez(path / "shard-000001.npz", **arrays)
    manifest = dict(rows=6, accepted=6, rejected=0, shards=2)
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert convert(path, 2)["verified"] == 6
    for file in path.glob("shard-*.npz"):
        with np.load(file) as data:
            assert set(data.files) == {"tokens", "labels", "cp", "mate", "split"}
            for name in data.files:
                expected = arrays[name][:, :69] if name == "tokens" else arrays[name]
                assert data[name].tobytes() == expected.tobytes()
    assert convert(path, 1)["bytes_saved"] == 0
    m = json.loads((path / "manifest.json").read_text())
    assert m["token_length"] == 69 and not m["position_keys"] and not m["legal_masks"]


def test_failed_write_preserves_original(tmp_path):
    path = tmp_path / "data"
    dataset(path)
    file = path / "shard-000000.npz"
    original = file.read_bytes()
    with patch("slim.np.savez", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            convert_shard(file)
    assert file.read_bytes() == original
    assert not list(path.glob("*.partial"))


def test_rust_slim(binary, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(rec(chess.STARTING_FEN, "e2e4")) + "\n")
    outputs = []
    for name, options in [("normal", []), ("slim", ["--slim"])]:
        output = tmp_path / name
        subprocess.run([str(binary), "--input", str(source), "--output", str(output),
                        "--vocab", str(ROOT / "moves.json"), *options], check=True)
        assert verify(output)["verified"] == 1
        with np.load(output / "shard-000000.npz") as data:
            outputs.append({key: data[key] for key in data.files})
    normal, slim = outputs
    assert set(slim) == {"tokens", "labels", "cp", "mate", "split"}
    for name, array in slim.items():
        expected = normal[name][:, :69] if name == "tokens" else normal[name]
        assert array.tobytes() == expected.tobytes()
    m = json.loads((tmp_path / "slim" / "manifest.json").read_text())
    assert m["token_length"] == 69 and not m["position_keys"]
    result = subprocess.run([str(binary), "--input", str(source), "--output",
                             str(tmp_path / "bad"), "--slim", "--masks"],
                            capture_output=True)
    assert result.returncode != 0
