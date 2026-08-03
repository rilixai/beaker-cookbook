"""Build RilixAI JSONL task-reference splits from the frozen LAB split lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.dataset import read_split


def build_dataset(output_dir: Path) -> None:
    """Write train/val rows that resolve to canonical public LAB tasks at run time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        task_ids = read_split(split)
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for task_id in task_ids:
                row = {"task_id": task_id, "source_commit": HARVEY_LABS_COMMIT}
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        counts[split] = len(task_ids)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "harveyai/harvey-labs",
                "source_commit": HARVEY_LABS_COMMIT,
                "splits": counts,
                "labels": "task.json rubric criteria loaded from the pinned public source",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to receive train.jsonl and val.jsonl")
    args = parser.parse_args()
    build_dataset(args.output_dir)


if __name__ == "__main__":
    main()
