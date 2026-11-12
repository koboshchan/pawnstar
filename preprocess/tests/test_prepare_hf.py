import hashlib
import json
from types import SimpleNamespace

import pytest
import prepare_hf


@pytest.fixture
def local_snapshot(monkeypatch, tmp_path):
    payload = b"local-test-parquet"
    path = tmp_path / "data" / "data_0000.parquet"
    path.parent.mkdir()
    path.write_bytes(payload)
    file = SimpleNamespace(
        rfilename="data/data_0000.parquet", size=len(payload),
        lfs=SimpleNamespace(sha256=hashlib.sha256(payload).hexdigest()),
    )
    class Api:
        def dataset_info(self, repo, revision, files_metadata):
            assert revision == prepare_hf.REVISION
            assert files_metadata is True
            return SimpleNamespace(siblings=[file])
    monkeypatch.setattr(prepare_hf, "HfApi", Api)
    monkeypatch.setattr(prepare_hf, "DIRECTORY", tmp_path)
    monkeypatch.setattr(prepare_hf.subprocess, "run",
                        lambda *a, **k: pytest.fail("must reuse existing local files"))
    return path


def test_local_reuse_and_provenance(local_snapshot):
    prepare_hf.main()
    proof = json.loads((prepare_hf.DIRECTORY / "pawnstar-source.json").read_text())
    assert proof["revision"] == prepare_hf.REVISION
    assert proof["files"][0]["sha256"] == hashlib.sha256(local_snapshot.read_bytes()).hexdigest()


def test_local_mismatch_not_overwritten(local_snapshot):
    local_snapshot.write_bytes(b"x" * local_snapshot.stat().st_size)
    with pytest.raises(SystemExit, match="SHA256 mismatch"):
        prepare_hf.main()
    assert local_snapshot.read_bytes().startswith(b"x")
    assert not (prepare_hf.DIRECTORY / "pawnstar-source.json").exists()


def test_unexpected_local_shard(local_snapshot):
    (local_snapshot.parent / "extra.parquet").write_bytes(b"extra")
    with pytest.raises(SystemExit, match="Unexpected parquet shards"):
        prepare_hf.main()
