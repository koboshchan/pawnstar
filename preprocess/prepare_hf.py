"""Reuse local parquet shards, download missing files fast, verify a pinned HF snapshot."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import HfApi

REPO = "Lichess/chess-position-evaluations"
REVISION = os.environ.get("CHESS_HF_REVISION", "8c5285259fdbf3218bf2582de9775dd1378f520b")
DIRECTORY = Path(os.environ.get("CHESS_HF_INPUT", "/content/lichess-hf"))


def main():
    if len(REVISION) != 40 or any(c not in "0123456789abcdef" for c in REVISION):
        raise SystemExit("CHESS_HF_REVISION must be an exact lowercase commit SHA")
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = sorted(
        (f for f in info.siblings if f.rfilename.startswith("data/") and f.rfilename.endswith(".parquet")),
        key=lambda f: f.rfilename,
    )
    if not files:
        raise SystemExit("No HF parquet shards")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    missing = [f.rfilename for f in files if not (DIRECTORY / f.rfilename).is_file()]
    if missing:
        required = sum(f.size for f in files if f.rfilename in missing)
        if shutil.disk_usage(DIRECTORY).free < required + 5_000_000_000:
            raise SystemExit("Insufficient free space for missing HF shards")
        # HF Hub v1+ uses parallel Xet transfers; hf_transfer is obsolete there.
        subprocess.run(
            ["hf", "download", REPO, "--repo-type", "dataset", "--revision", REVISION,
             "--local-dir", str(DIRECTORY), "--include", *missing],
            check=True,
            env={**os.environ, "HF_XET_HIGH_PERFORMANCE": "1"},
        )
    proof = []
    for f in files:
        path = DIRECTORY / f.rfilename
        if path.stat().st_size != f.size:
            raise SystemExit(f"Size mismatch, refusing to overwrite local file {path}")
        if not f.lfs:
            raise SystemExit(f"No authoritative SHA256 for {f.rfilename}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != f.lfs.sha256:
            raise SystemExit(f"SHA256 mismatch for {path}. Local dataset does not match pinned revision.")
        proof.append(dict(path=f.rfilename, bytes=f.size, sha256=digest))
        print(f"Verified {f.rfilename}", flush=True)
    expected = {f.rfilename for f in files}
    extra = {p.relative_to(DIRECTORY).as_posix() for p in (DIRECTORY / "data").glob("*.parquet")} - expected
    if extra:
        raise SystemExit(f"Unexpected parquet shards in data/: {sorted(extra)}")
    (DIRECTORY / "pawnstar-source.json").write_text(
        json.dumps(dict(repository=REPO, revision=REVISION, files=proof), indent=2) + "\n"
    )
    print(f"Using verified local snapshot at {DIRECTORY}. No upload started.")


if __name__ == "__main__":
    main()
