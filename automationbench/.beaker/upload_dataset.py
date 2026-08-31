"""Upload a tiny, domain-balanced AutomationBench dataset to Beaker."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from automationbench_skills.data import load_split


AGENT_KEY = "automationbench-skills"
DATASET_NAME = "automationbench-skills-quickstart"
DOMAIN_ORDER = ("sales", "marketing", "operations", "support", "finance", "hr")
TRAIN_COUNT = 7
VAL_COUNT = 3


def _select_samples(count: int) -> list[Any]:
    by_domain: dict[str, list[Any]] = defaultdict(list)
    for sample in load_split("train"):
        by_domain[sample.domain].append(sample)

    selected: list[Any] = []
    offset = 0
    while len(selected) < count:
        for domain in DOMAIN_ORDER:
            samples = by_domain[domain]
            if offset < len(samples):
                selected.append(samples[offset])
                if len(selected) == count:
                    return selected
        offset += 1
    return selected


def _row(sample: Any) -> dict[str, Any]:
    return {
        "id": sample.task_name,
        "input": {"task_name": sample.task_name},
        "expected": {
            "partial_credit": 1.0,
            "task_completed_correctly": 1.0,
        },
        "metadata": {
            "domain": sample.domain,
            "benchmark_split": "train",
            "source_index": sample.index,
        },
        "group_key": sample.domain,
    }


def _write_jsonl(path: Path, samples: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(_row(sample), sort_keys=True) + "\n")


def main() -> None:
    selected = _select_samples(TRAIN_COUNT + VAL_COUNT)
    train_samples = selected[:TRAIN_COUNT]
    val_samples = selected[TRAIN_COUNT:]

    with tempfile.TemporaryDirectory(prefix="beaker-automationbench-") as temp_dir:
        dataset_dir = Path(temp_dir)
        _write_jsonl(dataset_dir / "train.jsonl", train_samples)
        _write_jsonl(dataset_dir / "val.jsonl", val_samples)
        upload = subprocess.run(
            [
                "beaker",
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                DATASET_NAME,
                "--total-count",
                str(TRAIN_COUNT + VAL_COUNT),
                "--split",
                f"train={TRAIN_COUNT}",
                "--split",
                f"val={VAL_COUNT}",
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
