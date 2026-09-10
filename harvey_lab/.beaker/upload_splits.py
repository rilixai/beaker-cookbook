"""Convert frozen Harvey LAB splits to a temporary Beaker JSONL dataset and upload it.

Source of truth remains the pinned ``harveyai/harvey-labs`` task tree (fetched
via ``harvey_lab.data.fetch.ensure_task_dirs``). Generated JSONL is staged only
in an OS temp directory — never written into the repository or ``.beaker/``.

The frozen split lists are ordered round-robin across practice areas, so any
prefix stays distribution-representative. The first hosted experiment uses a
small prefix: LAB rollouts are long (up to 200 turns plus a batched rubric
judge).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harvey_lab.data.dataset import HarveyLabRecord, load_records, read_split
from harvey_lab.data.fetch import ensure_task_dirs


TRAIN_LIMIT = 8
TEST_LIMIT = 4
DATASET_NAME = "harvey-lab-quickstart"
AGENT_KEY = "harvey-lab-agent"


def record_to_row(record: HarveyLabRecord) -> dict[str, Any]:
    """Map one LAB task record to a Beaker standard JSONL row."""
    return {
        "id": record.task_id,
        "input": {
            "task_id": record.task_id,
            "practice_area": record.practice_area,
            "title": record.title,
            "work_type": record.work_type,
            "instructions": record.instructions,
            "deliverables": dict(record.deliverables),
            "documents": list(record.documents),
        },
        "expected": {
            "criteria": [
                {
                    "id": c.id,
                    "title": c.title,
                    "match_criteria": c.match_criteria,
                    "deliverables": list(c.deliverables),
                }
                for c in record.criteria
            ],
        },
        "metadata": {
            "task_fingerprint": record.task_fingerprint,
            "source": "harveyai/harvey-labs",
        },
        "group_key": record.practice_area,
    }


def materialize_splits(
    *,
    train_limit: int = TRAIN_LIMIT,
    test_limit: int = TEST_LIMIT,
    cache_dir: Path | None = None,
    tasks_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch real split prefixes and return ``(train_rows, test_rows)``."""
    train_ids = read_split("train")[: max(0, train_limit)]
    test_ids = read_split("test")[: max(0, test_limit)]
    if not train_ids or not test_ids:
        raise ValueError("Both train and test must include at least one real task id.")

    needed = list(dict.fromkeys([*train_ids, *test_ids]))
    root = tasks_root if tasks_root is not None else ensure_task_dirs(needed, cache_dir=cache_dir)
    by_id = {r.task_id: r for r in load_records(root, task_ids=needed)}
    missing = [tid for tid in needed if tid not in by_id]
    if missing:
        raise FileNotFoundError(f"Missing task.json for: {missing[:5]}")

    train_rows = [record_to_row(by_id[tid]) for tid in train_ids]
    test_rows = [record_to_row(by_id[tid]) for tid in test_ids]
    return train_rows, test_rows


def write_splits(
    dataset_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("test", test_rows)):
        path = dataset_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    train_rows, test_rows = materialize_splits()
    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        write_splits(dataset_dir, train_rows, test_rows)
        beaker = shutil.which("beaker")
        if not beaker:
            raise RuntimeError("beaker CLI is not on PATH")
        upload = subprocess.run(
            [
                beaker,
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                DATASET_NAME,
                "--agent",
                AGENT_KEY,
                "--total-count",
                str(len(train_rows) + len(test_rows)),
                "--split",
                f"train={len(train_rows)}",
                "--split",
                f"test={len(test_rows)}",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    print(upload.stdout)


if __name__ == "__main__":
    main()
