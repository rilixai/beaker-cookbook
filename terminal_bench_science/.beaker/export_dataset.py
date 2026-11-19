"""Export pinned public tasks without inventing a training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from benchmark_data import COMMIT, REPOSITORY, Row


def export(source: Path, destination: Path) -> dict[str, int]:
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("Source checkout must match the pinned commit")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD"], check=True)
    tracked = set(subprocess.check_output(["git", "-C", str(source), "ls-files"], text=True).splitlines())
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    resources = {}
    for task_id, directory, instruction, expected, metadata, extra_input in task_records(source):
        verifier_files = {}
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(source).as_posix()
            if relative not in tracked or not path.is_file() or path.is_symlink():
                continue
            if (
                path.suffix in {".py", ".sh", ".toml", ".yaml", ".json"}
                and "solution" not in path.parts
                and "authoring" not in path.parts
            ):
                verifier_files[path.relative_to(directory).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **({"text": path.read_text()} if path.suffix in {".py", ".sh", ".toml", ".yaml"} else {}),
                }
        if not verifier_files:
            raise ValueError(f"No public verifier files: {task_id}")
        expected["verifier_files"] = verifier_files
        row = Row.model_validate(
            {
                "id": task_id,
                "input": {"task_id": task_id, "instruction": instruction, "source_commit": COMMIT, **extra_input},
                "expected": expected,
                "metadata": metadata,
            }
        )
        rows.append(row.model_dump())
        resources[task_id] = {
            "repository": REPOSITORY,
            "commit": COMMIT,
            "directory": directory.relative_to(source).as_posix(),
        }
    if len(rows) != EXPECTED_COUNT or len({row["id"] for row in rows}) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} unique tasks, got {len(rows)}")
    (destination / "train.jsonl").write_text("")
    (destination / "test.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "source": REPOSITORY,
        "commit": COMMIT,
        "split_policy": "Full public evaluation set; no invented training split",
        "task_ids": {"train": [], "test": [row["id"] for row in rows]},
        "metadata": {"review_resources": resources},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"train": 0, "test": len(rows)}


EXPECTED_COUNT = 96


def task_records(source: Path) -> Iterator[tuple[str, Path, str, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    for config_path in sorted((source / "tasks").rglob("task.toml")):
        directory = config_path.parent
        raw_config = config_path.read_text()
        try:
            config = tomllib.loads(raw_config)
            config_error = None
        except tomllib.TOMLDecodeError as exc:
            config = {}
            config_error = str(exc)
        name = directory.relative_to(source / "tasks").as_posix()
        expected = {"reward": 1, "verifier_config": config.get("verifier", {})}
        metadata = {
            "domain": name.split("/")[0],
            "field": name.split("/")[1],
            "task_config": raw_config,
            "task_config_parse_error": config_error,
            "source_scope": "All 96 tasks on the pinned upstream main commit; not the older 70-task v0.1 release",
        }
        yield name, directory, (directory / "instruction.md").read_text(), expected, metadata, {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True, help="Use an OS temporary directory")
    args = parser.parse_args()
    print(json.dumps(export(args.source.resolve(), args.destination.resolve())))
