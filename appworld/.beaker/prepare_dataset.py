"""Upload complete source scenarios from train/dev using temporary JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path


def scenario_rows(data: Path, split: str, limit: int) -> list[dict]:
    groups: dict[str, list[str]] = OrderedDict()
    for task_id in (data / "datasets" / f"{split}.txt").read_text().splitlines():
        if task_id:
            groups.setdefault(task_id.rsplit("_", 1)[0], []).append(task_id)
    rows = []
    selected = list(groups.items())
    if limit:
        selected = selected[:limit]
    for scenario_id, task_ids in selected:
        tasks, requirements = [], {}
        for task_id in task_ids:
            task = data / "tasks" / task_id
            specs = json.loads((task / "specs.json").read_text())
            labels = json.loads((task / "ground_truth/test_data.json").read_text())
            tasks.append({"task_id": task_id, "instruction": specs["instruction"]})
            requirements[task_id] = [label["requirement"] for label in labels]
        rows.append({"id": scenario_id, "tasks": tasks, "requirements": requirements, "split": split})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scenarios", required=True, type=int, help="Scenarios per split; 0 means all")
    args = parser.parse_args()
    if args.scenarios < 0:
        parser.error("--scenarios must be nonnegative")
    project = Path(__file__).resolve().parents[1]
    cli = str(Path(sys.executable).parent / "beaker")
    env = dict(os.environ)
    rows = {
        "train": scenario_rows(project / "data", "train", args.scenarios),
        "test": scenario_rows(project / "data", "dev", args.scenarios),
    }
    if {r["id"] for r in rows["train"]} & {r["id"] for r in rows["test"]}:
        raise ValueError("Train/dev scenario overlap")
    with tempfile.TemporaryDirectory(prefix="beaker-appworld-dataset-") as temp:
        for split, values in rows.items():
            with (Path(temp) / f"{split}.jsonl").open("w") as output:
                for row in values:
                    output.write(json.dumps(row) + "\n")
        subprocess.run(
            [
                cli,
                "--config-file",
                ".beaker/beaker.yaml",
                "run",
                "smoke",
                "--strict",
                "--integration-id",
                "appworld_openai_agents_sdk",
                "--config",
                json.dumps({"local_dataset_path": temp}),
            ],
            cwd=project,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                cli,
                "dataset",
                "upload",
                temp,
                "--name",
                args.name,
                "--total-count",
                str(sum(map(len, rows.values()))),
                "--split",
                f"train={len(rows['train'])}",
                "--split",
                f"test={len(rows['test'])}",
                "--agent",
                args.agent,
                "--json",
            ],
            cwd=project,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
