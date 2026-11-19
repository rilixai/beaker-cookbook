"""Rows from the frozen ALE v1.1 public task selection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPOSITORY = "rdi-berkeley/agents-last-exam"
COMMIT = "2e91edd3cd26baa35a38127c90c8e5241dfbe62f"


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    instruction: str = Field(min_length=1)
    source_commit: str
    input_files: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def pinned(self) -> TaskInput:
        parts = self.task_id.split("/")
        if len(parts) != 2 or any(not p or p in {".", ".."} for p in parts) or "\\" in self.task_id:
            raise ValueError("Expected domain/task ID")
        if self.source_commit != COMMIT:
            raise ValueError("Task must use the pinned ALE source commit")
        return self


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: TaskInput
    expected: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self) -> Row:
        if self.id != self.input.task_id or not self.expected.get("verifier_files"):
            raise ValueError("Task ID and public grader source are required")
        return self
