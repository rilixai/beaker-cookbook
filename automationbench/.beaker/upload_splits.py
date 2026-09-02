"""Convert frozen AutomationBench splits to a temporary Beaker JSONL dataset and upload it.

Writes nothing into the repository. Invoke from the automationbench project root.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from automationbench_skills.data.tasks import PUBLIC_DOMAINS, load_split


TRAIN_PER_DOMAIN = 4
TEST_PER_DOMAIN = 2
DATASET_NAME = "automationbench-skills-quickstart"


def _take_per_domain(split: str, per_domain: int) -> list[dict[str, object]]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for sample in load_split(split):
        by_domain[sample.domain].append(sample.task_name)
    rows: list[dict[str, object]] = []
    for domain in PUBLIC_DOMAINS:
        for task_name in by_domain[domain][:per_domain]:
            rows.append(
                {
                    "id": task_name,
                    "input": {"task_name": task_name},
                    "expected": {},
                    "metadata": {"domain": domain, "source_split": split},
                    "group_key": domain,
                }
            )
    return rows


def main() -> None:
    train_rows = _take_per_domain("train", TRAIN_PER_DOMAIN)
    test_rows = _take_per_domain("test", TEST_PER_DOMAIN)
    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        splits = {"train": train_rows, "test": test_rows}
        for split_name, rows in splits.items():
            split_path = dataset_dir / f"{split_name}.jsonl"
            with split_path.open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
        beaker = shutil.which("beaker")
        if not beaker:
            raise RuntimeError("beaker CLI is not on PATH")
        upload = subprocess.run(
            [
                beaker,
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                DATASET_NAME,
                "--agent",
                "automationbench-skills",
                "--total-count",
                str(sum(len(rows) for rows in splits.values())),
                "--split",
                f"train={len(train_rows)}",
                "--split",
                f"test={len(test_rows)}",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    print(upload.stdout)


if __name__ == "__main__":
    main()
