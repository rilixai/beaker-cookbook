"""Convert the frozen Harvey splits to temporary JSONL, validate, and upload.

Run from harvey_lab/. --smoke-only validates real tasks without an upload or
model calls. Generated JSONL is removed when this command exits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from beaker_integration import TaskRow, row_payload

from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.dataset import load_records, read_split
from harvey_lab.data.fetch import ensure_task_dirs


BEAKER_YAML = Path(__file__).with_name("beaker.yaml")
INTEGRATION_ID = "harvey-lab"


def configured_agent_key() -> str:
    config = yaml.safe_load(BEAKER_YAML.read_text(encoding="utf-8"))
    key = config["integrations"][INTEGRATION_ID].get("agent_key")
    if not key:
        raise ValueError("Run `beaker agent setup` first or supply --agent.")
    return str(key)


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", help="defaults to agent_key in .beaker/beaker.yaml")
    parser.add_argument("--name", default="harvey-lab-quickstart")
    parser.add_argument("--train-limit", type=positive_count, default=8)
    parser.add_argument("--test-limit", type=positive_count, default=4)
    parser.add_argument("--full", action="store_true", help="use all 1660 train and 100 test tasks")
    parser.add_argument("--tasks-root", type=Path, help="tasks/ from a checkout at HARVEY_LABS_COMMIT")
    parser.add_argument("--smoke-only", action="store_true", help="validate locally without uploading")
    args = parser.parse_args()
    beaker = shutil.which("beaker")
    if beaker is None:
        raise RuntimeError("beaker CLI is not on PATH; run with `uv run python .beaker/upload_splits.py`")
    splits = {
        split: read_split(split)[: None if args.full else limit]
        for split, limit in (("train", args.train_limit), ("test", args.test_limit))
    }
    ids = [task_id for task_ids in splits.values() for task_id in task_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("frozen train and test splits overlap")
    tasks_root = args.tasks_root or ensure_task_dirs(ids)
    with tempfile.TemporaryDirectory(prefix="harvey-beaker-dataset-") as directory:
        dataset_dir = Path(directory)
        for split, task_ids in splits.items():
            with (dataset_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for record in load_records(tasks_root, task_ids=task_ids):
                    task_input, expected = row_payload(record)
                    row = TaskRow(
                        id=record.task_id,
                        input=task_input,
                        expected=expected,
                        metadata={"practice_area": record.practice_area, "source_split": split},
                    )
                    stream.write(row.model_dump_json() + "\n")
        (dataset_dir / "manifest.json").write_text(
            json.dumps({"source": "harveyai/harvey-labs", "commit": HARVEY_LABS_COMMIT}), encoding="utf-8"
        )
        subprocess.run(
            [
                beaker,
                "run",
                "smoke",
                "--strict",
                "--integration-id",
                INTEGRATION_ID,
                "--config",
                json.dumps(
                    {"local_dataset_path": str(dataset_dir), "extra": {"tasks_root": str(tasks_root.resolve())}}
                ),
            ],
            check=True,
        )
        if args.smoke_only:
            return
        agent_key = args.agent or configured_agent_key()
        upload = subprocess.run(
            [
                beaker,
                "dataset",
                "upload",
                str(dataset_dir),
                "--name",
                args.name,
                "--agent",
                agent_key,
                "--total-count",
                str(len(ids)),
                "--split",
                f"train={len(splits['train'])}",
                "--split",
                f"test={len(splits['test'])}",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        artifact = json.loads(upload.stdout)
        selector = f"{artifact['artifact_key']}@{artifact['dataset_revision']}"
    print(f"Dataset: {selector}")
    subprocess.run(
        [
            beaker,
            "run",
            "smoke",
            "--strict",
            "--integration-id",
            INTEGRATION_ID,
            "--agent",
            agent_key,
            "--dataset",
            selector,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
