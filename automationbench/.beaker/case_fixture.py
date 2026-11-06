"""One frozen case format, executed through the pinned PI API runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
from pydantic import BaseModel, ConfigDict, Field

from automationbench_skills.data.tasks import Sample


PI_COMMIT = "858718b19e5d66c74451432783d34ee64a550a15"
SOURCE_COMMITS = {"original": "4a8e1061254004d9dac807054eed33fad7d1ff14", "pi": PI_COMMIT}


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["original", "pi"]
    fallback_time: str
    tool_fault: Literal["empty_email_lookup", "drop_opportunity_amount"] | None = None
    trace_capture: Literal["complete", "incomplete"] = "complete"


class TaskInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(min_length=1)
    initial_state: dict[str, Any]
    assertions: list[dict[str, Any]]
    zapier_tools: list[str]


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: int
    prompt: list[dict[str, str]]
    answer: str
    info: TaskInfo


class CaseFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Task
    scenario: Scenario

    def validate_reference(self, task_name: str, expected: dict, metadata: dict) -> None:
        if self.task.info.task_name != task_name:
            raise ValueError("input.task_name disagrees with runtime_fixture")
        if self.task.info.assertions != expected.get("assertions"):
            raise ValueError("expected.assertions disagrees with runtime_fixture")
        if metadata.get("benchmark_commit") != SOURCE_COMMITS[self.scenario.source]:
            raise ValueError("benchmark_commit must identify the fixture's source revision")
        for name in ("task", "scenario"):
            value = getattr(self, name).model_dump(mode="json")
            digest = hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
            if metadata.get(f"{name}_sha256") != digest:
                raise ValueError(f"{name}_sha256 disagrees with runtime_fixture")
        WorldState.model_validate(self.initial_state())
        for assertion in self.task.info.assertions:
            if assertion.get("type") not in AssertionRegistry._handlers:
                raise ValueError(f"Unknown PI assertion type: {assertion.get('type')}")

    def initial_state(self) -> dict[str, Any]:
        initial = self.task.info.model_dump(mode="json")["initial_state"]
        initial.setdefault("meta", {}).setdefault("current_time", self.scenario.fallback_time)
        return initial

    def sample(self) -> Sample:
        info = self.task.info.model_dump(mode="json")
        info["initial_state"] = self.initial_state()
        info["_case_scenario"] = self.scenario.model_dump(mode="json")
        return Sample(
            task_name=self.task.info.task_name,
            domain=self.task.info.task_name.split(".")[0],
            index=self.task.example_id,
            prompt=self.task.prompt,
            answer=self.task.answer,
            info=info,
        )
