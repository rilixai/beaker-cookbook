"""Balanced demo selection and export must preserve source split boundaries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction

import pytest

from automationbench_skills.data import demo
from automationbench_skills.data.tasks import PUBLIC_DOMAINS, load_split, read_split_names, task_family


@pytest.mark.parametrize("split,per_domain", [("train", 20), ("test", 10)])
def test_balanced_demo_preserves_domain_and_family_proportions(split, per_domain):
    selected = demo.select_demo(split)
    source = load_split(split)
    assert Counter(s.domain for s in selected) == dict.fromkeys(PUBLIC_DOMAINS, per_domain)
    selected_ids = {s.task_name for s in selected}
    assert len(selected_ids) == len(selected)
    assert selected_ids <= {s.task_name for s in source}
    assert not selected_ids & set(read_split_names("excluded_cases"))
    assert [s.task_name for s in selected] == [s.task_name for s in source if s.task_name in selected_ids]
    for domain in PUBLIC_DOMAINS:
        pool = Counter(task_family(s.task_name) for s in source if s.domain == domain)
        actual = Counter(task_family(s.task_name) for s in selected if s.domain == domain)
        for family, count in pool.items():
            target = Fraction(per_domain * count, sum(pool.values()))
            assert abs(actual[family] - target) < 1
    assert [s.task_name for s in demo.select_demo(split)] == [s.task_name for s in selected]


def test_local_export_is_reproducible_disjoint_and_has_matching_manifest(tmp_path, monkeypatch):
    import subprocess

    def forbid_upload(*args, **kwargs):
        raise AssertionError("Local demo generation must not invoke a subprocess or uploader")

    monkeypatch.setattr(subprocess, "run", forbid_upload)
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = demo.write_demo(first)
    assert demo.write_demo(second) == report
    assert report["split_counts"] == {"train": 140, "test": 70}
    assert report["case_count"] == 210
    assert report["excluded_case_count"] == 25
    assert report["selected_excluded_cases"] == report["train_test_overlap"] == []
    ids = {}
    for split in ("train", "test"):
        content = (first / f"{split}.jsonl").read_bytes()
        assert content == (second / f"{split}.jsonl").read_bytes()
        assert hashlib.sha256(content).hexdigest() == report["sha256"][f"{split}.jsonl"]
        rows = [json.loads(line) for line in content.splitlines()]
        ids[split] = {row["id"] for row in rows}
        assert rows == demo.demo_rows(split)
        assert all(row["metadata"]["benchmark_commit"] == demo.BENCHMARK_COMMIT for row in rows)
    assert not ids["train"] & ids["test"]
    assert json.loads((first / "verification.json").read_text()) == report
    with pytest.raises(FileExistsError):
        demo.write_demo(first)


def test_insufficient_domain_cannot_silently_unbalance_demo(monkeypatch):
    samples = load_split("train")
    monkeypatch.setattr(demo, "load_split", lambda _: [s for s in samples if s.domain != "hr"])
    with pytest.raises(ValueError, match="train/hr needs 20 eligible tasks, but has 0"):
        demo.select_demo("train")
