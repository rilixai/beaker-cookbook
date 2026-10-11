"""Frozen split invariants and generator determinism."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from officeqa import splits
from officeqa.data.dataset import EvalRecord, load_records_from_csv, read_split_ids, select_split
from officeqa.splits import generate_split, largest_remainder, n_files_bucket, stratum_of


SPLIT_DIR = Path(splits.__file__).parent


def test_frozen_sizes_disjoint_and_cover_133() -> None:
    train, test = read_split_ids("train"), read_split_ids("test")
    assert len(train) == 93 and len(test) == 40
    assert len(set(train)) == 93 and len(set(test)) == 40
    assert not set(train) & set(test)
    assert len(set(train) | set(test)) == 133
    assert (SPLIT_DIR / "README.md").is_file()


def test_split_files_are_plain_uid_lines() -> None:
    for name in ("train.txt", "test.txt"):
        text = (SPLIT_DIR / name).read_text(encoding="utf-8")
        assert text.endswith("\n") and all(line and line == line.strip() for line in text.splitlines())


def test_features(records: list[EvalRecord]) -> None:
    assert [n_files_bucket(n) for n in (1, 2, 3, 7)] == ["1", "2", "3+", "3+"]
    assert stratum_of(records[0]) == ("1", 1980)
    assert stratum_of(records[1]) == ("2", 2010)
    assert stratum_of(records[2]) == ("3+", 1940)


def test_largest_remainder() -> None:
    assert largest_remainder([50, 30, 20], 10) == [5, 3, 2]
    assert sum(largest_remainder([7, 7, 7, 7, 5], 13)) == 13
    assert largest_remainder([1, 1, 1], 1) == [1, 0, 0]  # ties broken by position, deterministically


def test_generator_is_deterministic_and_exact(records: list[EvalRecord]) -> None:
    # 3 records / 3 strata; ask for 1 test id.
    a = generate_split(records, n_test=1)
    b = generate_split(records, n_test=1)
    assert a.train == b.train and a.test == b.test
    assert len(a.test) == 1 and len(a.train) == 2 and not set(a.train) & set(a.test)
    assert set(a.train) | set(a.test) == {r.uid for r in records}
    c = generate_split(records, n_test=1, seed=1)
    assert set(c.train) | set(c.test) == {r.uid for r in records}


def test_select_split_filters_and_orders(records: list[EvalRecord], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("officeqa.data.dataset.read_split_ids", lambda split: [records[2].uid, records[0].uid])
    chosen = select_split(records, "train")
    assert [r.uid for r in chosen] == [records[2].uid, records[0].uid]


@pytest.mark.skipif(
    not os.environ.get("OFFICEQA_PRO_CSV"),
    reason="set OFFICEQA_PRO_CSV=<path to officeqa_pro.csv> to check regeneration",
)
def test_frozen_split_matches_regeneration_from_real_csv() -> None:
    records = load_records_from_csv(Path(os.environ["OFFICEQA_PRO_CSV"]))
    assert len(records) == 133
    result = generate_split(records)
    assert list(result.train) == read_split_ids("train")
    assert list(result.test) == read_split_ids("test")
