"""Export Harvey LAB task rows from a local checkout to a rilixai dataset.

Each JSONL line is one normalized task row — the shape
``HarveyLabDataLoader.parse_row`` expects: ``task_id``, ``practice_area``,
``title``, ``work_type``, ``instructions``, ``deliverables``, ``documents``
(the sorted document filenames — fetched at run time from the pinned commit
by the workspace's GitHub task source), and ``criteria``. Rows are split by
*practice area* so no area leaks between train, val, and the optional test
split (the recipe stratifies on ``practice_area``). Pass ``--test-areas N``
to carve off a disjoint ``test.jsonl`` the optimizer scores the winning
candidate on after optimization (unbiased held-out number).

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
from collections import defaultdict
from pathlib import Path

from harvey_lab.data.dataset import HarveyLabRecord, attach_document_blobs, load_harvey_lab_records, record_to_row


def _cap_round_robin(rows: list[HarveyLabRecord], limit: int | None) -> list[HarveyLabRecord]:
    """Subsample ``rows`` to at most ``limit`` while spreading across practice
    areas (round-robin) so every area stays represented. Deterministic:
    preserves the loader's within-area order and iterates areas alphabetically.
    """
    if limit is None or len(rows) <= limit:
        return rows
    by_area: dict[str, list[HarveyLabRecord]] = defaultdict(list)
    for r in rows:
        by_area[r.practice_area].append(r)
    order = sorted(by_area)
    picked: list[HarveyLabRecord] = []
    idx = 0
    while len(picked) < limit:
        progressed = False
        for area in order:
            bucket = by_area[area]
            if idx < len(bucket):
                picked.append(bucket[idx])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", required=True, help="Path to a local harvey-labs checkout's tasks/ dir.")
    ap.add_argument("--out", default="scripts/_datasets/harvey_lab")
    ap.add_argument("--practice-areas", default=None, help="Comma-separated practice areas (default: all).")
    ap.add_argument("--max-per-area", type=int, default=None, help="Optional cap on tasks per practice area.")
    ap.add_argument("--val-areas", type=int, default=3, help="Number of practice areas to hold out for validation.")
    ap.add_argument(
        "--test-areas",
        type=int,
        default=0,
        help="Number of practice areas to hold out for the optional post-optimization test split (disjoint from train/val).",
    )
    ap.add_argument("--max-train", type=int, default=None, help="Cap total train rows (round-robin across areas).")
    ap.add_argument("--max-val", type=int, default=None, help="Cap total val rows (round-robin across areas).")
    ap.add_argument("--max-test", type=int, default=None, help="Cap total test rows (round-robin across areas).")
    ap.add_argument(
        "--embed-documents",
        action="store_true",
        help="Base64-embed each task's documents into the JSONL rows so the dataset is "
        "self-contained (no run-time fetch from raw.githubusercontent.com).",
    )
    args = ap.parse_args()

    areas = [a.strip() for a in args.practice_areas.split(",")] if args.practice_areas else None
    records = load_harvey_lab_records(args.tasks_root, practice_areas=areas, max_per_area=args.max_per_area)
    if not records:
        raise SystemExit(f"No task records found under {args.tasks_root!r}.")

    present_areas = sorted({r.practice_area for r in records})
    if len(present_areas) <= args.val_areas + args.test_areas:
        raise SystemExit(
            f"Only {len(present_areas)} practice areas; cannot hold out "
            f"{args.val_areas} val + {args.test_areas} test areas and keep a train split."
        )
    # Carve disjoint tail groups: last `test_areas` for test, the next
    # `val_areas` for validation, the remainder for train — no area leaks.
    test_areas = set(present_areas[len(present_areas) - args.test_areas :]) if args.test_areas else set()
    val_end = len(present_areas) - args.test_areas
    val_areas = set(present_areas[val_end - args.val_areas : val_end])

    train = [r for r in records if r.practice_area not in val_areas and r.practice_area not in test_areas]
    val = [r for r in records if r.practice_area in val_areas]
    test = [r for r in records if r.practice_area in test_areas]

    # Cap each split (round-robin across its areas) before embedding, so we only
    # base64-encode the documents we actually keep.
    train = _cap_round_robin(train, args.max_train)
    val = _cap_round_robin(val, args.max_val)
    test = _cap_round_robin(test, args.max_test)

    if args.embed_documents:
        train = [attach_document_blobs(r, args.tasks_root) for r in train]
        val = [attach_document_blobs(r, args.tasks_root) for r in val]
        test = [attach_document_blobs(r, args.tasks_root) for r in test]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, rows: list) -> None:
        with (out / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(record_to_row(r), ensure_ascii=False) + "\n")

    _dump("train.jsonl", train)
    _dump("val.jsonl", val)
    if args.test_areas:
        _dump("test.jsonl", test)

    print(f"practice_areas={len(present_areas)} (val={sorted(val_areas)} test={sorted(test_areas)})")
    msg = f"wrote {len(train)} train, {len(val)} val"
    if args.test_areas:
        msg += f", {len(test)} test"
    print(f"{msg} rows to {out}")


if __name__ == "__main__":
    main()
