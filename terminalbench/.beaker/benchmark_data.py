"""Pinned benchmark tasks, with full evaluation and custom 8/4 development datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import requests
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, model_validator


COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
ARCHIVE_URL = f"https://codeload.github.com/harbor-framework/terminal-bench-2/tar.gz/{COMMIT}"
ARCHIVE_SHA256 = "afcd2b9813fc08e8d2b955db8165cc057acf0e37f74e55d26fc2506da196cebe"
TRAIN = (
    "kv-store-grpc",
    "pytorch-model-recovery",
    "nginx-request-logging",
    "prove-plus-comm",
    "hf-model-inference",
    "fix-git",
    "configure-git-webserver",
    "cobol-modernization",
)
TEST = ("crack-7z-hash", "raman-fitting", "openssl-selfsigned-cert", "polyglot-c-py")
TASKS = TRAIN + TEST


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    instruction: str


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: TaskInput
    expected: dict[str, int]

    @model_validator(mode="after")
    def validate_task(self) -> Row:
        if self.id != self.input.task_id or self.id not in full_task_ids():
            raise ValueError("Row must identify one task from the pinned TerminalBench 2.0 release")
        if self.expected != {"reward": 1}:
            raise ValueError("TerminalBench expected reward must be 1")
        return self


def task_root() -> Path:
    """Download once under an OS temporary directory; verify before extraction."""
    cache = Path(tempfile.gettempdir()) / f"beaker-terminalbench-{COMMIT}"
    cache.mkdir(exist_ok=True)
    root = cache / f"terminal-bench-2-{COMMIT}"
    with FileLock(str(cache / "download.lock")):
        if (cache / "complete").exists():
            return root
        with tempfile.TemporaryDirectory(dir=cache) as scratch:
            archive = Path(scratch) / "tasks.tar.gz"
            digest = hashlib.sha256()
            with requests.get(ARCHIVE_URL, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with archive.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != ARCHIVE_SHA256:
                raise ValueError("TerminalBench archive checksum mismatch")
            with tarfile.open(archive) as bundle:
                bundle.extractall(scratch, filter="data")
            if root.exists():
                shutil.rmtree(root)
            shutil.move(str(Path(scratch) / root.name), root)
        (cache / "complete").touch()
    return root


def full_task_ids() -> tuple[str, ...]:
    names = tuple(
        sorted(
            path.name
            for path in task_root().iterdir()
            if (path / "task.toml").is_file() and (path / "instruction.md").is_file()
        )
    )
    if len(names) != 89:
        raise ValueError(f"Expected 89 tasks in the pinned benchmark, found {len(names)}")
    return names


def rows(root: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        Row(
            id=name,
            input=TaskInput(task_id=name, instruction=(root / name / "instruction.md").read_text()),
            expected={"reward": 1},
        ).model_dump()
        for name in names
    ]


def write_dataset(destination: Path, *, full: bool = False) -> dict[str, int]:
    root = task_root()
    destination.mkdir(parents=True, exist_ok=True)
    splits = {"train": (), "test": full_task_ids()} if full else {"train": TRAIN, "test": TEST}
    for split, names in splits.items():
        text = "".join(json.dumps(row) + "\n" for row in rows(root, names))
        (destination / f"{split}.jsonl").write_text(text)
    return {split: len(names) for split, names in splits.items()}


def stage_task(task_id: str, destination: Path) -> Path:
    if task_id not in full_task_ids():
        raise ValueError(f"Task outside the pinned benchmark: {task_id}")
    source = task_root() / task_id
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "task.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        # Never stage the oracle solution in the candidate's files.
        for name in ("task.toml", "instruction.md", "environment", "tests"):
            bundle.add(source / name, arcname=name)
    return archive
