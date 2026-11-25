"""Preparation: checksum, label agreement, stratified deterministic splits, category filter, option cap."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from mmdm import data
from mmdm.data import SPLIT_FILES, convert, load_rows, make_splits, parse_categories, prepare, source_file


def record(index: int, category: str, semantics: str = "categorical_hard", options: int = 3) -> dict[str, Any]:
    """Synthetic rows test splitting only; they are never evaluation data."""
    gold = [1.0] + [0.0] * (options - 1)
    task = {"set_hard": "multichoice", "stochastic_ordinal_score": "score"}.get(semantics, "choice")
    if semantics == "set_hard":
        gold = [0.5, 0.5] + [0.0] * (options - 2)
    return {
        "id": index,
        "state": f"synthetic state {index}",
        "question": "Which option?",
        "options": [{"id": f"o{i}", "name": f"Option {i}", "criteria": None} for i in range(options)],
        "task": task,
        "score_values": list(range(options)) if task == "score" else None,
        "gold": gold,
        "answer": {"kind": "hard", "probabilities": list(gold)},
        "correct_option_ids": ["o0", "o1"] if semantics == "set_hard" else ["o0"],
        "label_semantics": semantics,
        "task_category": category,
        "category": category,
        "task_subcategory": "synthetic",
        "source": "mmdm",
    }


def synthetic(count_per_stratum: int = 50) -> list[dict[str, Any]]:
    records = []
    strata = [
        ("intent_routing", "categorical_hard"),
        ("intent_routing", "set_hard"),
        ("score_rubric", "categorical_hard"),
    ]
    for category, semantics in strata:
        for _ in range(count_per_stratum):
            records.append(record(len(records) + 1, category, semantics))
    return records


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    (tmp_path / data.FILENAME).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        source_file(tmp_path)


def test_gold_and_answer_disagreeing_fails() -> None:
    records = synthetic(2)
    records[3]["answer"]["probabilities"] = [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="disagree on 1 rows"):
        convert(records)


def test_splits_are_disjoint_stratified_and_deterministic() -> None:
    rows = convert(synthetic(50))
    splits = make_splits(rows)
    assert splits == make_splits(list(reversed(rows)))
    ids = [{r.id for r in splits[name]} for name in ("train", "validation", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(s) for s in ids) == len(rows)
    for name, share in (("train", 30), ("validation", 10), ("test", 10)):
        strata = Counter((r.expected.task_category, r.expected.label_semantics) for r in splits[name])
        assert set(strata.values()) == {share}, (name, strata)
    # Ordered by a seeded hash, not by ID, so any prefix mixes strata.
    order = [r.id for r in splits["validation"]]
    assert order == sorted(order, key=lambda i: hashlib.sha256(f"mmdm-v1:order:{i}".encode()).hexdigest())
    assert order != sorted(order)


def test_category_filter_and_option_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = synthetic(50)
    records.append(record(len(records) + 1, "intent_routing", options=data.OPTION_CAP + 1))
    content = "".join(json.dumps(r) + "\n" for r in records).encode()
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / data.FILENAME).write_bytes(content)
    monkeypatch.setattr(data, "SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(data, "UPSTREAM_ROWS", len(records))

    full = prepare(tmp_path / "full", tmp_path / "cache")
    subset = prepare(tmp_path / "subset", tmp_path / "cache", parse_categories("intent_routing"))
    assert full["counts"] == {"train": 90, "validation": 30, "test": 30}
    assert subset["counts"] == {"train": 60, "validation": 20, "test": 20}
    for name, path in SPLIT_FILES.items():
        subset_rows = load_rows(tmp_path / "subset" / path)
        assert {r.expected.task_category for r in subset_rows} == {"intent_routing"}, name
        # A subset gets exactly the assignment the full dataset gives those rows.
        full_ids = [r.id for r in load_rows(tmp_path / "full" / path) if r.expected.task_category == "intent_routing"]
        assert [r.id for r in subset_rows] == full_ids
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    assert manifest["dropped_over_option_cap"]["ids"] == [f"mmdm-{len(records)}"]
    assert manifest["dataset_revision"] == data.REVISION and manifest["split_seed"] == "mmdm-v1"
    with pytest.raises(ValueError, match="Unknown MMDM categories"):
        parse_categories("intent_routing,not_a_category")


def test_rows_keep_labels_and_category_out_of_input() -> None:
    row = convert([record(7, "intent_routing")])[0]
    assert row.id == "mmdm-7"
    assert set(row.input.model_dump()) == {"state", "question", "options", "task", "score_values"}
    assert set(row.expected.model_dump()) == {
        "label_semantics",
        "probabilities",
        "correct_option_ids",
        "task_category",
        "task_subcategory",
        "source",
    }
