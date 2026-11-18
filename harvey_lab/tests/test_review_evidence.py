"""Uploaded review evidence retains source bytes and makes extraction gaps explicit."""

import gzip
import hashlib
import json
import zipfile

import pytest

from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.review_evidence import package_documents


def test_packaging_retains_originals_and_is_reproducible(tmp_path):
    task = tmp_path / "task"
    docs = task / "documents"
    docs.mkdir(parents=True)
    source = b"Payment is due in thirty days."
    (docs / "contract.txt").write_bytes(source)
    output = tmp_path / "dataset"
    output.mkdir()
    entry = package_documents(task, "contracts/example", output)
    packed = (output / entry["artifact_path"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == entry["sha256"]
    raw = gzip.decompress(packed)
    assert len(raw) == entry["decoded_bytes"]
    payload = json.loads(raw)
    assert payload["source_commit"] == HARVEY_LABS_COMMIT
    assert payload["documents"]["contract.txt"]["text"] == source.decode()
    assert payload["documents"]["contract.txt"]["sha256"] == hashlib.sha256(source).hexdigest()
    with zipfile.ZipFile(output / entry["archive_path"]) as archive:
        assert archive.read("contract.txt") == source
    assert package_documents(task, "contracts/example", output) == entry


def test_unsupported_and_truncated_files_are_not_silently_complete(tmp_path, monkeypatch):
    from harvey_lab.data import review_evidence

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "scan.png").write_bytes(b"image")
    (docs / "brief.pdf").write_bytes(b"pdf")
    monkeypatch.setattr(review_evidence, "extract_text", lambda _: "first page\n... (truncated pages)")
    entry = package_documents(tmp_path, "example", tmp_path)
    documents = json.loads(gzip.decompress((tmp_path / entry["artifact_path"]).read_bytes()))["documents"]
    assert documents["scan.png"]["status"] == "unsupported"
    assert documents["brief.pdf"]["status"] == "partial"
    assert documents["brief.pdf"]["limitations"]


def test_source_symlinks_are_rejected(tmp_path):
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents/link").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symbolic links"):
        package_documents(tmp_path, "example", tmp_path)


def test_source_directory_symlink_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "documents").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        package_documents(task, "example", tmp_path)
