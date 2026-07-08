"""Export Harvey LAB task rows from a local checkout to a rilixai dataset.

Each JSONL line is one normalized task row — the shape
``HarveyLabDataLoader.parse_row`` expects: ``task_id``, ``practice_area``,
``title``, ``work_type``, ``instructions``, ``deliverables``, ``documents``
(the sorted document filenames — fetched at run time from the pinned commit
by the workspace's GitHub task source), and ``criteria``. Rows are split by
*practice area* so no area leaks between train and val (the recipe
stratifies on ``practice_area``).

Clone the benchmark first (documents are NOT copied into the dataset, only
their filenames — the hosted run fetches them from the pinned commit):

    git clone https://github.com/harveyai/harvey-labs
    uv run python scripts/export_harvey_lab_dataset.py --tasks-root harvey-labs/tasks

Then upload:

    rilixai dataset upload --name harvey-lab-dataset scripts/_datasets/harvey_lab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvey_lab.data.dataset import load_harvey_lab_records, record_to_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", required=True, help="Path to a local harvey-labs checkout's tasks/ dir.")
    ap.add_argument("--out", default="scripts/_datasets/harvey_lab")
    ap.add_argument("--practice-areas", default=None, help="Comma-separated practice areas (default: all).")
    ap.add_argument("--max-per-area", type=int, default=None, help="Optional cap on tasks per practice area.")
    ap.add_argument("--val-areas", type=int, default=3, help="Number of practice areas to hold out for validation.")
    args = ap.parse_args()

    areas = [a.strip() for a in args.practice_areas.split(",")] if args.practice_areas else None
    records = load_harvey_lab_records(args.tasks_root, practice_areas=areas, max_per_area=args.max_per_area)
    if not records:
        raise SystemExit(f"No task records found under {args.tasks_root!r}.")

    present_areas = sorted({r.practice_area for r in records})
    if len(present_areas) <= args.val_areas:
        raise SystemExit(f"Only {len(present_areas)} practice areas; cannot hold out {args.val_areas}.")
    val_areas = set(present_areas[-args.val_areas :])

    train = [r for r in records if r.practice_area not in val_areas]
    val = [r for r in records if r.practice_area in val_areas]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(record_to_row(r), ensure_ascii=False) + "\n")
    with (out / "val.jsonl").open("w", encoding="utf-8") as fh:
        for r in val:
            fh.write(json.dumps(record_to_row(r), ensure_ascii=False) + "\n")

    print(f"practice_areas={len(present_areas)} (val={sorted(val_areas)})")
    print(f"wrote {len(train)} train, {len(val)} val rows to {out}")


if __name__ == "__main__":
    main()
