"""Upload the native Toolathlon quickstart dataset splits without persisting local JSONL."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from toolathlon_setup import build_quickstart_row, prepare_toolathlon


TRAIN_TASKS = ("find-alita-paper", "excel-market-research")
TEST_TASKS = ("interview-report",)


def main() -> None:
    _, source = prepare_toolathlon()
    splits = {
        "train": [build_quickstart_row(source, task) for task in TRAIN_TASKS],
        "test": [build_quickstart_row(source, task) for task in TEST_TASKS],
    }
    if {row["id"] for row in splits["train"]} & {row["id"] for row in splits["test"]}:
        raise ValueError("Training and evaluation tasks must be disjoint")

    with tempfile.TemporaryDirectory(prefix="beaker-toolathlon-dataset-") as temporary:
        directory = Path(temporary)
        for split, selected in splits.items():
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in selected),
                encoding="utf-8",
            )
        subprocess.run(
            [
                "beaker",
                "run",
                "smoke",
                "--strict",
                "--integration-id",
                "toolathlon_baseline",
                "--config",
                json.dumps({"local_dataset_path": temporary}),
            ],
            check=True,
        )
        subprocess.run(
            [
                "beaker",
                "dataset",
                "upload",
                temporary,
                "--agent",
                "toolathlon",
                "--name",
                "toolathlon-quickstart",
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
