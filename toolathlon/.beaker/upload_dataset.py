"""Upload Toolathlon finalpool tasks as a Beaker dataset (one row per task)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from toolathlon_setup import TASKS, required_env


# Toolathlon ships one pool without splits; hold out a stable quarter for test.
TEST_FRACTION = 0.25


def rows(*, no_secrets: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in TASKS.iterdir() if path.is_dir()):
        if no_secrets and required_env(task_dir.name):
            continue
        prompt = (task_dir / "docs" / "task.md").read_text()
        selected.append({"id": task_dir.name, "input": {"task": task_dir.name, "prompt": prompt}, "expected": {}})
    return selected


def split(task: str) -> str:
    bucket = int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < TEST_FRACTION else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-secrets", action="store_true", help="Only tasks that need no account or API key.")
    parser.add_argument("--name", default=None, help="Dataset name (defaults to toolathlon or toolathlon-no-secrets).")
    args = parser.parse_args()
    agent = "toolathlon-default-no-secrets" if args.no_secrets else "toolathlon-default"
    with tempfile.TemporaryDirectory(prefix="beaker-toolathlon-dataset-") as temporary:
        directory = Path(temporary)
        selected = rows(no_secrets=args.no_secrets)
        counts = {}
        for name in ("train", "test"):
            subset = [row for row in selected if split(str(row["id"])) == name]
            counts[name] = len(subset)
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in subset))
        subprocess.run(
            [
                "beaker",
                "dataset",
                "upload",
                temporary,
                "--agent",
                agent,
                "--name",
                args.name or agent.removesuffix("-default").replace("-default-", "-"),
                "--total-count",
                str(len(selected)),
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
