"""Export real HARVEY-LABS task records into Beaker's JSONL dataset layout.

Usage:
    .venv/bin/python scripts/export_beaker_dataset.py \
        --tasks-root /path/to/harvey-labs/tasks --output-dir /path/to/dataset \
        --train-limit 20 --val-limit 10

The source task JSON and its document fingerprint are copied as ground truth;
this script never manufactures cases or labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.dataset import HarveyLabRecord, load_records, read_split


def _row(record: HarveyLabRecord) -> dict[str, Any]:
    return {
        "id": record.task_id,
        "input": {"task_id": record.task_id},
        "expected": {
            "task_fingerprint": record.task_fingerprint,
            "title": record.title,
            "instructions": record.instructions,
            "deliverables": dict(record.deliverables),
            "criteria": [
                {
                    "id": criterion.id,
                    "title": criterion.title,
                    "match_criteria": criterion.match_criteria,
                    "deliverables": list(criterion.deliverables),
                }
                for criterion in record.criteria
            ],
        },
        "metadata": {"practice_area": record.practice_area, "work_type": record.work_type},
        "group_key": record.practice_area,
    }


def _write_split(path: Path, records: list[HarveyLabRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_row(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite populated dataset directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = read_split("train")[: args.train_limit]
    val_ids = read_split("val")[: args.val_limit]
    train_records = load_records(args.tasks_root, task_ids=train_ids)
    val_records = load_records(args.tasks_root, task_ids=val_ids)
    _write_split(args.output_dir / "train.jsonl", train_records)
    _write_split(args.output_dir / "val.jsonl", val_records)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "harveyai/harvey-labs",
                "commit": HARVEY_LABS_COMMIT,
                "train_count": len(train_records),
                "val_count": len(val_records),
                "format": "harvey-lab-beaker-v1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
