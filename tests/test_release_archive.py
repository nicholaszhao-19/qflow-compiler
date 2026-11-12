"""Regression checks for manifest-controlled source archives."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_release import release_paths
from smoke_release import verify_and_extract


def test_repack_uses_only_manifest_paths(tmp_path):
    data = b"release input"
    (tmp_path / "README.md").write_bytes(data)
    (tmp_path / "personal-draft.txt").write_text("untracked")
    (tmp_path / "release_manifest.json").write_text(json.dumps({"files": [{"path": "README.md"}]}))
    assert release_paths(tmp_path) == [tmp_path / "README.md"]


def test_repack_rejects_private_report_even_if_listed_in_manifest(tmp_path):
    report = tmp_path / "report" / "draft.txt"
    report.parent.mkdir()
    report.write_text("private")
    (tmp_path / "release_manifest.json").write_text(json.dumps({"files": [{"path": "report/draft.txt"}]}))
    with pytest.raises(ValueError, match="Private report path"):
        release_paths(tmp_path)


def test_archive_verifies_hashes_before_extraction(tmp_path):
    data = b"release input"
    path = tmp_path / "candidate.zip"
    manifest = {"files": [{"path": "README.md", "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("qflow-compiler/release_manifest.json", json.dumps(manifest))
        archive.writestr("qflow-compiler/README.md", b"changed input")
    output = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Release digest or size differs"):
        verify_and_extract(path, output)
    assert not output.exists()
