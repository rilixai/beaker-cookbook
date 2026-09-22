"""Check the real quickstart selection without uploading a dataset."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from automationbench_skills.data import PUBLIC_DOMAINS, load_split


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("upload_splits", ROOT / ".beaker/upload_splits.py")
assert SPEC is not None and SPEC.loader is not None
upload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upload)


def test_quickstart_excludes_audited_cases_and_preserves_quotas() -> None:
    excluded = {
        line.strip()
        for line in (ROOT / ".beaker/excluded_cases.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert len(excluded) == 114
    original = {split: load_split(split) for split in ("train", "test")}
    assert excluded <= {sample.task_name for samples in original.values() for sample in samples}
    selected: dict[str, set[str]] = {}
    for split, per_domain in (("train", 6), ("test", 3)):
        rows = upload._take_per_domain(split, per_domain)
        ids = [row["id"] for row in rows]
        selected[split] = set(ids)
        assert len(ids) == len(set(ids)) == 6 * per_domain
        assert not set(ids) & excluded
        assert set(ids) <= {sample.task_name for sample in original[split]}
        assert Counter(row["group_key"] for row in rows) == dict.fromkeys(PUBLIC_DOMAINS, per_domain)
        for domain in PUBLIC_DOMAINS:
            eligible = [s for s in original[split] if s.domain == domain and s.task_name not in excluded]
            assert [row["id"] for row in rows if row["group_key"] == domain] == [
                s.task_name for s in eligible[:per_domain]
            ]
        assert rows == upload._take_per_domain(split, per_domain)
    assert selected["train"].isdisjoint(selected["test"])


def test_quickstart_fails_if_a_domain_cannot_fill_its_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload, "load_split", lambda split: [])
    with pytest.raises(ValueError, match="Not enough eligible train cases"):
        upload._take_per_domain("train", 6)
