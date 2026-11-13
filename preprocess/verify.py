"""Validate original and slim token shard datasets."""
import argparse
import json
import pathlib

import numpy as np


def validate_shard(data):
    size = len(data["labels"])
    tokens = data["tokens"]
    assert tokens.dtype == np.uint8
    assert tokens.ndim == 2 and tokens.shape[0] == size
    assert tokens.shape[1] in (69, 128)
    if tokens.shape[1] == 128:
        assert not tokens[:, 69:].any()
    for name in ("labels", "cp", "mate", "split"):
        assert data[name].shape == (size,), name
    assert ((data["labels"] >= 0) & (data["labels"] < 1968)).all()
    assert np.isin(data["split"], (0, 1, 2)).all()
    if "legal" in data.files:
        assert data["legal"].shape == (size, 246)
        masks = np.unpackbits(data["legal"], axis=1, bitorder="little")
        assert masks[np.arange(size), data["labels"]].all()
    if "keys" in data.files:
        assert data["keys"].shape == (size,)
        keys = data["keys"] % 1000
        expected = np.where(keys < 980, 0, np.where(keys < 990, 1, 2))
        assert np.array_equal(data["split"], expected)
    return size


def verify(path):
    path = pathlib.Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    files = sorted(path.glob("shard-*.npz"))
    count = 0
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            count += validate_shard(data)
    assert len(files) == manifest["shards"]
    assert count == manifest["accepted"]
    assert manifest["rows"] == manifest["accepted"] + manifest["rejected"]
    return {"verified": count, "shards": len(files),
            "bytes": sum(f.stat().st_size for f in path.iterdir() if f.is_file())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory)))
