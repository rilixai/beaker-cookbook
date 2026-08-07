"""Convert real Harvey LAB tasks into temporary Beaker JSONL splits.

Source of truth remains the pinned ``harveyai/harvey-labs`` task tree (fetched
via ``harvey_lab.data.fetch.ensure_task_dirs``). Generated JSONL is staged only
in an OS temp directory — never written into the repository or ``.beaker/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harvey_lab.data.dataset import HarveyLabRecord, load_records, read_split
from harvey_lab.data.fetch import ensure_task_dirs


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
    train_limit: int,
    val_limit: int,
    cache_dir: Path | None = None,
    tasks_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch real split prefixes and return ``(train_rows, val_rows)``."""
    train_ids = read_split("train")[: max(0, train_limit)]
    val_ids = read_split("val")[: max(0, val_limit)]
    if not train_ids or not val_ids:
        raise ValueError("Both train and val must include at least one real task id.")

    needed = list(dict.fromkeys([*train_ids, *val_ids]))
    if tasks_root is None:
        root = ensure_task_dirs(needed, cache_dir=cache_dir)
    else:
        root = tasks_root

    by_id = {r.task_id: r for r in load_records(root, task_ids=needed)}
    missing = [tid for tid in needed if tid not in by_id]
    if missing:
        raise FileNotFoundError(f"Missing task.json for: {missing[:5]}")

    train_rows = [record_to_row(by_id[tid]) for tid in train_ids]
    val_rows = [record_to_row(by_id[tid]) for tid in val_ids]
    return train_rows, val_rows


def write_splits(
    dataset_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        path = dataset_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=2, help="Real train tasks to include.")
    parser.add_argument("--val-limit", type=int, default=1, help="Real val tasks to include.")
    parser.add_argument("--tasks-root", type=Path, default=None, help="Optional local harvey-labs tasks/.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional fetch cache directory.")
    parser.add_argument(
        "--upload-name",
        default=None,
        help="If set, upload the staged JSONL via `beaker dataset upload` before cleanup.",
    )
    parser.add_argument("--agent", default=None, help="Agent key for upload (optional).")
    parser.add_argument(
        "--print-dir",
        action="store_true",
        help="Print the temp dataset dir and keep it until Enter (for local dry-run).",
    )
    args = parser.parse_args(argv)

    train_rows, val_rows = materialize_splits(
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        cache_dir=args.cache_dir,
        tasks_root=args.tasks_root,
    )

    with tempfile.TemporaryDirectory(prefix="beaker-harvey-lab-") as temp_dir:
        dataset_dir = Path(temp_dir)
        write_splits(dataset_dir, train_rows, val_rows)
        print(
            f"Staged {len(train_rows)} train + {len(val_rows)} val rows under {dataset_dir}",
            file=sys.stderr,
        )

        if args.upload_name:
            cmd = [
                "beaker",
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                args.upload_name,
                "--total-count",
                str(len(train_rows) + len(val_rows)),
                "--split",
                f"train={len(train_rows)}",
                "--split",
                f"val={len(val_rows)}",
            ]
            if args.agent:
                cmd.extend(["--agent", args.agent])
            subprocess.run(cmd, check=True)
            return 0

        if args.print_dir:
            print(dataset_dir)
            try:
                input("Press Enter after dry-run to delete the temporary dataset… ")
            except EOFError:
                pass
            return 0

        print(
            "No --upload-name or --print-dir; dataset was staged and discarded. "
            "Pass --print-dir for local dry-run or --upload-name after a READY spec.",
            file=sys.stderr,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
