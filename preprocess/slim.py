"""Convert shards in place to 69-token arrays without keys or legal masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from verify import validate_shard, verify


def convert_shard(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        validate_shard(data)
        if data["tokens"].shape[1] == 69 and set(data.files) == {
            "tokens", "labels", "cp", "mate", "split"
        }:
            return 0
        arrays = {name: data[name] for name in ("labels", "cp", "mate", "split")}
        arrays["tokens"] = np.ascontiguousarray(data["tokens"][:, :69])
    before = path.stat().st_size
    # A same-directory temporary and atomic replace keep interrupted shards intact.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".partial", delete=False) as stream:
            temp = Path(stream.name)
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temp, allow_pickle=False) as data:
            validate_shard(data)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return before - path.stat().st_size


def convert(directory, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    directory = Path(directory)
    # Validate everything before replacing any files; mixed widths are resumable.
    verify(directory)
    files = sorted(directory.glob("shard-*.npz"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        saved = sum(pool.map(convert_shard, files))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(token_length=69, active_tokens=69, legal_masks=False,
                    position_keys=False)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory,
                                         prefix="manifest.", suffix=".partial",
                                         delete=False) as stream:
            temp = Path(stream.name)
            stream.write(json.dumps(manifest, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, manifest_path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    result = verify(directory)
    result["bytes_saved"] = saved
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    print(json.dumps(convert(args.directory, args.workers)))
