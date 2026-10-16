"""Upload a small AutomationBench dataset (20 train / 20 test) to Beaker.

Rows are the benchmark's own public tasks from the frozen, filtered splits:
per split, the first cases of each domain in split order, spread across the
six domains. Generated JSONL is written only to a temporary directory.

    uv run python .beaker/upload_dataset.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from automationbench_skills.data.tasks import PUBLIC_DOMAINS, Sample, load_split


DATASET_NAME = "automationbench-20x20"
PER_SPLIT = 20
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pick(samples: list[Sample], total: int) -> list[Sample]:
    base, extra = divmod(total, len(PUBLIC_DOMAINS))
    picked: list[Sample] = []
    for i, domain in enumerate(PUBLIC_DOMAINS):
        quota = base + (1 if i < extra else 0)
        picked.extend([s for s in samples if s.domain == domain][:quota])
    return picked


def _row(sample: Sample) -> dict:
    instruction = next(m["content"] for m in reversed(sample.prompt) if m.get("role") == "user")
    return {
        "id": sample.task_name,
        "task_name": sample.task_name,
        "domain": sample.domain,
        "instruction": instruction,
        "assertions": sample.info["assertions"],
    }


def main() -> None:
    splits = {name: [_row(s) for s in _pick(load_split(name), PER_SPLIT)] for name in ("train", "test")}
    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        for name, rows in splits.items():
            with (dataset_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
        command = [
            "beaker",
            "dataset",
            "upload",
            str(dataset_dir),
            "--name",
            DATASET_NAME,
            "--total-count",
            str(sum(len(rows) for rows in splits.values())),
            *(arg for name, rows in splits.items() for arg in ("--split", f"{name}={len(rows)}")),
            "--json",
        ]
        # Run from the project root so the CLI selects this project's .beaker/beaker.yaml and agent.
        upload = subprocess.run(command, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    artifact = json.loads(upload.stdout)
    print(json.dumps(artifact, indent=2))
    print(f"\nDataset selector: {artifact['artifact_key']}@{artifact['dataset_revision']}", file=sys.stderr)


if __name__ == "__main__":
    main()
