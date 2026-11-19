"""Pinned public task rows and isolated task bundles for Harbor."""

from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPOSITORY = "harbor-framework/terminal-bench-science"
COMMIT = "d4f6c91be7eba71a9fc279c3809317489a8dd374"


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    instruction: str = Field(min_length=1)
    source_commit: str

    @model_validator(mode="after")
    def pinned(self) -> TaskInput:
        parts = self.task_id.split("/")
        if len(parts) != 3 or any(not p or p in {".", ".."} for p in parts) or "\\" in self.task_id:
            raise ValueError("Expected domain/field/task ID")
        if self.source_commit != COMMIT:
            raise ValueError("Task must use the pinned source commit")
        return self


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: TaskInput
    expected: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self) -> Row:
        if self.id != self.input.task_id or self.expected.get("reward") != 1:
            raise ValueError("Task ID and expected full-credit reward must match")
        if not self.expected.get("verifier_files"):
            raise ValueError("Public verifier source is required")
        return self


def stage_task(task_id: str, destination: Path) -> Path:
    """Fetch only the selected task; never put oracle solutions in the agent bundle."""
    TaskInput(task_id=task_id, instruction="validated", source_commit=COMMIT)
    cache = Path(tempfile.gettempdir()) / f"beaker-science-{COMMIT}"
    cache.mkdir(exist_ok=True)
    checkout = cache / "source"
    archive = destination / "task.tar"
    with FileLock(str(cache / "fetch.lock")):
        if not (checkout / ".git").exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    f"https://github.com/{REPOSITORY}.git",
                    str(checkout),
                ],
                check=True,
            )
        subprocess.run(["git", "-C", str(checkout), "fetch", "origin", COMMIT], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "archive",
                "--format=tar",
                f"{COMMIT}:tasks/{task_id}",
                "task.toml",
                "instruction.md",
                "environment",
                "tests",
                "-o",
                str(archive),
            ],
            check=True,
        )
    with tarfile.open(archive) as bundle:
        if any(member.issym() or member.islnk() for member in bundle.getmembers()):
            raise ValueError("Task bundles cannot contain links")
    return archive
