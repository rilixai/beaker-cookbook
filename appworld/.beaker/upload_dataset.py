"""Upload four complete scenarios from the official training split."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

from appworld_setup import prepare_appworld


def main() -> None:
    root = prepare_appworld()
    groups: dict[str, list[str]] = OrderedDict()
    for task_id in (root / "data/datasets/train.txt").read_text().splitlines():
        groups.setdefault(task_id.split("_")[0], []).append(task_id)
    rows = []
    for scenario, tasks in list(groups.items())[:4]:
        inputs, expected = [], {}
        for task_id in tasks:
            task_dir = root / "data/tasks" / task_id
            specs = json.loads((task_dir / "specs.json").read_text())
            inputs.append({"task_id": task_id, "instruction": specs["instruction"]})
            expected[task_id] = json.loads((task_dir / "ground_truth/test_data.json").read_text())
        rows.append({"id": scenario, "input": {"tasks": inputs}, "expected": expected})
    with tempfile.TemporaryDirectory(prefix="beaker-appworld-dataset-") as temporary:
        directory = Path(temporary)
        for split, selected in (("train", rows[:3]), ("test", rows[3:])):
            (directory / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        subprocess.run(
            [
                "beaker",
                "run",
                "smoke",
                "--strict",
                "--integration-id",
                "appworld_openai_agents_sdk",
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
                "appworld-fresh",
                "--name",
                "appworld-sgc-quickstart",
                "--total-count",
                "4",
                "--split",
                "train=3",
                "--split",
                "test=1",
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
