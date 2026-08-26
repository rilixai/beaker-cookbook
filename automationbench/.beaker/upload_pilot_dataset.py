from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from automationbench_skills import load_split
from automationbench_skills.data.tasks import Sample


DATASET_NAME = "automationbench-skills-pilot"
AGENT_KEY = "automationbench-skills"
SPLITS = {"train": load_split("train")[:10], "val": load_split("test")[:5]}


def _row(sample: Sample) -> dict[str, Any]:
    return {
        "id": sample.task_name,
        "input": {"task_name": sample.task_name},
        "expected": {"partial_credit": 1.0, "task_completed_correctly": 1.0},
        "metadata": {"domain": sample.domain},
        "group_key": sample.domain,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="beaker-automationbench-pilot-") as temp_dir:
        dataset_dir = Path(temp_dir)
        for split_name, samples in SPLITS.items():
            with (dataset_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as output:
                for sample in samples:
                    output.write(json.dumps(_row(sample)) + "\n")
        upload = subprocess.run(
            [
                "beaker",
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                DATASET_NAME,
                "--total-count",
                str(sum(len(samples) for samples in SPLITS.values())),
                "--split",
                f"train={len(SPLITS['train'])}",
                "--split",
                f"val={len(SPLITS['val'])}",
                "--agent",
                AGENT_KEY,
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    print(upload.stdout.strip())


if __name__ == "__main__":
    main()
