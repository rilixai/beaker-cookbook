"""Offline checks for training selection and the uploader's selection safeguards."""

from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from harvey_lab.data.dataset import read_split


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".beaker"))
    return importlib.import_module("sample_splits")


@pytest.fixture
def uploader(sampler: ModuleType) -> ModuleType:
    return importlib.import_module("upload_splits")


def area_counts(task_ids: list[str]) -> Counter[str]:
    return Counter(task_id.split("/", 1)[0] for task_id in task_ids)


def test_default_sample_matches_frozen_test_distribution(sampler: ModuleType) -> None:
    train, test = read_split("train"), read_split("test")
    selected = sampler.sample_train(train, test, 100, 0)

    assert len(selected) == len(set(selected)) == 100
    assert set(selected) <= set(train)
    assert set(selected).isdisjoint(test)
    assert len(area_counts(test)) == 26
    assert area_counts(selected) == area_counts(test)
    assert selected == sampler.sample_train(train, test, 100, 0)
    assert selected == sampler.sample_train(train[::-1], test[::-1], 100, 0)
    assert selected != sampler.sample_train(train, test, 100, 1)


@pytest.mark.parametrize("count", [1, 2, 26, 99, 101, 200])
def test_sample_quotas_stay_within_one_task(sampler: ModuleType, count: int) -> None:
    train, test = read_split("train"), read_split("test")
    selected = sampler.sample_train(train, test, count, 0)

    assert len(selected) == len(set(selected)) == count
    assert set(selected) <= set(train)
    counts = area_counts(selected)
    for area, test_count in area_counts(test).items():
        assert abs(counts[area] - count * test_count / len(test)) < 1


def test_largest_remainder_ties_use_area_name(sampler: ModuleType) -> None:
    train = [f"{area}/{i}" for area in ("a", "b", "c") for i in range(4)]
    test = ["a/test1", "a/test2", "b/test", "c/test"]

    # Quotas are 1.5, 0.75, 0.75; the two remaining slots go to b and c.
    assert area_counts(sampler.sample_train(train, test, 3, 0)) == {"a": 1, "b": 1, "c": 1}
    # At size two, b wins the equal 0.5 remainders over c.
    assert area_counts(sampler.sample_train(train, test, 2, 0)) == {"a": 1, "b": 1}


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        ([], ["a/test"], "must both contain tasks"),
        (["a/train"], [], "must both contain tasks"),
        (["a/train", "a/train"], ["a/test"], "duplicate task IDs"),
        (["a/train"], ["a/test", "a/test"], "duplicate task IDs"),
        (["a/shared"], ["a/shared"], "overlap"),
    ],
)
def test_sample_rejects_invalid_splits(sampler: ModuleType, train: list[str], test: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sampler.sample_train(train, test, 1, 0)


@pytest.mark.parametrize("count", [-1, 0, 3])
def test_sample_rejects_invalid_sizes(sampler: ModuleType, count: int) -> None:
    with pytest.raises(ValueError, match="train size must be between 1 and 2"):
        sampler.sample_train(["a/1", "a/2"], ["a/test"], count, 0)


def test_insufficient_capacity_does_not_redistribute_quotas(sampler: ModuleType) -> None:
    with pytest.raises(ValueError, match="a: need 2 train tasks.*only 1"):
        sampler.sample_train(["a/1", "b/1", "b/2"], ["a/test1", "a/test2", "b/test"], 3, 0)


def test_distribution_report_includes_areas_missing_from_training(sampler: ModuleType) -> None:
    assert sampler.distribution_report(["a/1", "a/2"], ["a/test", "b/test"]) == {
        "a": {"train_count": 2, "test_count": 1, "train_share": 1.0, "test_share": 0.5},
        "b": {"train_count": 0, "test_count": 1, "train_share": 0.0, "test_share": 0.5},
    }


@pytest.mark.parametrize("full", [False, True])
def test_uploader_dry_run_preserves_test_set_without_external_work(
    uploader: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], full: bool
) -> None:
    monkeypatch.setattr(sys, "argv", ["upload_splits.py", "--dry-run", *(["--full"] if full else [])])
    monkeypatch.setattr(uploader.shutil, "which", lambda _: pytest.fail("dry-run must not require the CLI"))
    monkeypatch.setattr(uploader, "ensure_task_dirs", lambda _: pytest.fail("dry-run must not download tasks"))
    uploader.main()
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["task_ids"]["test"] == read_split("test")
    assert manifest["sampling"]["test_truncated_for_smoke"] is False
    if full:
        assert manifest["task_ids"]["train"] == read_split("train")
        assert manifest["sampling"]["method"] == "full"
        assert manifest["sampling"]["seed"] is None
    else:
        assert len(manifest["task_ids"]["train"]) == 100
        assert manifest["sampling"]["method"] == "practice-area-largest-remainder-v1"
        assert manifest["sampling"]["seed"] == 0
        assert all(row["train_count"] == row["test_count"] for row in manifest["distribution"].values())


@pytest.mark.parametrize("extra_args", [[], ["--full"], ["--smoke-only", "--full"]])
def test_uploader_rejects_test_truncation_outside_smoke(
    uploader: ModuleType, monkeypatch: pytest.MonkeyPatch, extra_args: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["upload_splits.py", "--test-limit", "1", *extra_args])
    monkeypatch.setattr(uploader, "read_split", lambda _: pytest.fail("invalid options must fail before loading"))
    with pytest.raises(SystemExit) as exc:
        uploader.main()
    assert exc.value.code == 2


def test_full_mode_still_validates_splits(uploader: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["upload_splits.py", "--full", "--dry-run"])
    monkeypatch.setattr(uploader, "read_split", lambda _: ["a/shared"])
    with pytest.raises(ValueError, match="overlap"):
        uploader.main()
