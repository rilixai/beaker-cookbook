"""Upload official AppWorld train/dev rows without persisting generated JSONL."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from appworld_resources import prepare_resources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true", help="Upload 4 train and 3 dev tasks for a seven-task trial")
    args = parser.parse_args()
    root = prepare_resources()
    counts = {}
    with tempfile.TemporaryDirectory(prefix="beaker-appworld-dataset-") as temp:
        for upstream_split, beaker_split in (("train", "train"), ("dev", "test")):
            task_ids = (root / "data" / "datasets" / f"{upstream_split}.txt").read_text().splitlines()
            if args.small:
                task_ids = task_ids[: 4 if upstream_split == "train" else 3]
            counts[beaker_split] = len(task_ids)
            with (Path(temp) / f"{beaker_split}.jsonl").open("w") as output:
                for task_id in task_ids:
                    task = root / "data" / "tasks" / task_id
                    specs = json.loads((task / "specs.json").read_text())
                    assertions = json.loads((task / "ground_truth" / "test_data.json").read_text())
                    row = {
                        "id": task_id,
                        "instruction": specs["instruction"],
                        "requirements": [item["requirement"] for item in assertions],
                        "split": upstream_split,
                        "data_version": "0.2.0",
                    }
                    output.write(json.dumps(row) + "\n")
        subprocess.run(
            [
                str(Path(sys.executable).with_name("beaker")),
                "--config-file",
                ".beaker/beaker.yaml",
                "dataset",
                "upload",
                temp,
                "--name",
                "appworld-smoke-7" if args.small else "appworld-train-dev",
                "--agent",
                "appworld",
                "--total-count",
                str(sum(counts.values())),
                "--split",
                f"train={counts['train']}",
                "--split",
                f"test={counts['test']}",
                "--json",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
