"""Convert frozen AutomationBench split files into Beaker JSONL rows.

Writes only to a caller-supplied directory (an OS temp dir during upload).
Does not invent labels: each row is one real task from the frozen splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from automationbench_skills.data import load_split


def row_from_sample(sample: Any, *, split: str) -> dict[str, Any]:
    return {
        "id": sample.task_name,
        "input": {"task_name": sample.task_name},
        "expected": {},
        "metadata": {"domain": sample.domain, "split": split},
        "group_key": sample.domain,
    }


def write_splits(dataset_dir: Path) -> dict[str, int]:
    """Write train/val/test JSONL. Val is the train set: this recipe has no
    validation split and selects on train; test stays held out."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [row_from_sample(s, split="train") for s in load_split("train")]
    test_rows = [row_from_sample(s, split="test") for s in load_split("test")]
    splits = {
        "train": train_rows,
        "val": train_rows,
        "test": test_rows,
    }
    for name, rows in splits.items():
        path = dataset_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return {name: len(rows) for name, rows in splits.items()}
