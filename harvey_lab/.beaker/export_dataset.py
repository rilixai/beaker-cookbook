"""Export Harvey LAB's frozen task splits as Beaker JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.dataset import read_split


def export_dataset(output_dir: Path, *, train_limit: int | None = None, val_limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = {"train": train_limit, "val": val_limit, "test": None}
    counts: dict[str, int] = {}
    for split, limit in limits.items():
        task_ids = read_split(split)
        if limit is not None:
            task_ids = task_ids[:limit]
        rows = [
            {
                "id": task_id,
                "input": {"task_id": task_id},
                "expected": {"all_pass": 1.0, "criterion_pass_rate": 1.0},
                "metadata": {"split": split, "harvey_labs_commit": HARVEY_LABS_COMMIT},
                "group_key": task_id.partition("/")[0],
            }
            for task_id in task_ids
        ]
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        counts[split] = len(rows)

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "harveyai/harvey-labs",
                "commit": HARVEY_LABS_COMMIT,
                "splits": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    args = parser.parse_args()
    export_dataset(args.output_dir, train_limit=args.train_limit, val_limit=args.val_limit)


if __name__ == "__main__":
    main()
