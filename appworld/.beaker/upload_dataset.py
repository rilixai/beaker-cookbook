"""Upload complete scenarios from the official training and development splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

from appworld_setup import prepare_appworld


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tgc", action="store_true", help="Upload all 147 tasks individually for the TGC agent.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Upload all training scenarios and use dev for evaluation (default; kept for compatibility).",
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Upload a small set instead: the first 4 train scenarios (3 train, 1 held out).",
    )
    args = parser.parse_args()
    if args.quickstart and args.tgc:
        parser.error("--quickstart is not supported with --tgc")
    quick = args.quickstart
    cli = ["beaker", "--config-file", ".beaker/tgc.yaml"] if args.tgc else ["beaker"]
    integration_id = "appworld_tgc" if args.tgc else "app_world"
    root = prepare_appworld()
    splits = {}
    for source_split in ("train",) if quick else ("train", "dev"):
        groups: dict[str, list[str]] = OrderedDict()
        for task_id in (root / f"data/datasets/{source_split}.txt").read_text().splitlines():
            groups.setdefault(task_id if args.tgc else task_id.split("_")[0], []).append(task_id)
        rows = []
        selected_groups = list(groups.items())[:4] if quick else list(groups.items())
        for scenario, tasks in selected_groups:
            inputs, expected = [], {}
            for task_id in tasks:
                task_dir = root / "data/tasks" / task_id
                specs = json.loads((task_dir / "specs.json").read_text())
                inputs.append({"task_id": task_id, "instruction": specs["instruction"]})
                expected[task_id] = json.loads((task_dir / "ground_truth/test_data.json").read_text())
            if args.tgc:
                rows.append({"id": scenario, "input": inputs[0], "expected": expected[scenario]})
            else:
                rows.append({"id": scenario, "input": {"tasks": inputs}, "expected": expected})
        if quick:
            splits = {"train": rows[:3], "test": rows[3:]}
        else:
            splits["train" if source_split == "train" else "test"] = rows
    if {row["id"] for row in splits["train"]} & {row["id"] for row in splits["test"]}:
        raise ValueError("Training and evaluation scenarios must be disjoint")
    with tempfile.TemporaryDirectory(prefix="beaker-appworld-dataset-") as temporary:
        directory = Path(temporary)
        for split, selected in splits.items():
            (directory / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        subprocess.run(
            [
                *cli,
                "run",
                "smoke",
                "--strict",
                "--integration-id",
                integration_id,
                "--config",
                json.dumps({"local_dataset_path": temporary}),
            ],
            check=True,
        )
        subprocess.run(
            [
                *cli,
                "dataset",
                "upload",
                temporary,
                "--agent",
                "appworld-task-goal-completion" if args.tgc else "appworld-fresh",
                "--name",
                "appworld-tgc-full" if args.tgc else "appworld-sgc-quickstart" if quick else "appworld-sgc-full",
                "--total-count",
                str(sum(len(rows) for rows in splits.values())),
                "--split",
                f"train={len(splits['train'])}",
                "--split",
                f"test={len(splits['test'])}",
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
