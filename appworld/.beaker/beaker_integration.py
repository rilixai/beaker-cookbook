"""Optimize scenario goal completion using complete AppWorld train/dev scenarios."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from appworld_setup import prepare_appworld
from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    RepositoryRunSetup,
    RepositoryRunSetupResult,
    repository,
)
from pydantic import BaseModel, Field, model_validator


class TaskInput(BaseModel):
    task_id: str = Field(pattern=r"^[a-z0-9]+_[0-9]+$")
    instruction: str = Field(min_length=1)


class ScenarioInput(BaseModel):
    tasks: list[TaskInput] = Field(min_length=1)


class Row(BaseModel):
    id: str
    input: ScenarioInput
    expected: dict

    @model_validator(mode="after")
    def complete_scenario(self):
        task_ids = [task.task_id for task in self.input.tasks]
        if len(set(task_ids)) != len(task_ids) or any(t.split("_")[0] != self.id for t in task_ids):
            raise ValueError("A row must contain unique variants of one scenario")
        return self


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(self, *, runtime):
        self.root = prepare_appworld()
        yield RepositoryRunSetupResult()

    async def load_cases(self, row: Row, *, runtime) -> AsyncIterator[Case]:
        import json

        allowed_tasks = [
            task_id
            for split in ("train", "dev")
            for task_id in (self.root / f"data/datasets/{split}.txt").read_text().splitlines()
        ]
        expected_ids = {t for t in allowed_tasks if t.split("_")[0] == row.id}
        if {t.task_id for t in row.input.tasks} != expected_ids or not expected_ids:
            raise ValueError(f"Scenario {row.id} must include all its train/dev variants")
        for task in row.input.tasks:
            specs = json.loads((self.root / "data/tasks" / task.task_id / "specs.json").read_text())
            if task.instruction != specs["instruction"]:
                raise ValueError(f"Instruction mismatch for {task.task_id}")
            requirements = json.loads(
                (self.root / "data/tasks" / task.task_id / "ground_truth/test_data.json").read_text()
            )
            if row.expected.get(task.task_id) != requirements:
                raise ValueError(f"Ground-truth requirements mismatch for {task.task_id}")
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected)


async def run_case(*, case_input, runtime) -> CaseResult:
    from appworld_subprocess import run_isolated_case

    return await run_isolated_case(case_input=case_input, runtime=runtime)


async def score_case(*, case, result, case_files_dir: Path) -> CaseScore:
    tasks = result.output["tasks"]
    expected_ids = {task["task_id"] for task in case.input["tasks"]}
    if set(tasks) != expected_ids:
        raise ValueError("The evaluator did not return every scenario variant")
    checks = []
    for task_id, outcome in tasks.items():
        for passed, entries in ((True, outcome["passes"]), (False, outcome["failures"])):
            for entry in entries:
                checks.append(
                    Check(
                        name=entry["requirement"],
                        group=task_id,
                        verdict="pass" if passed else "fail",
                        message=entry.get("trace"),
                    )
                )
    successes = [bool(outcome["success"]) for outcome in tasks.values()]
    sgc = float(all(successes))
    tgc = sum(successes) / len(successes)
    combined = 0.8 * tgc + 0.2 * sgc
    return CaseScore(
        objective=combined,
        field_scores={
            "combined_goal_completion": combined,
            "scenario_goal_completion": sgc,
            "task_goal_completion": tgc,
        },
        checks=tuple(checks),
    )


integration = Integration(
    targets=repository(("src/appworld_openai_agents_sdk/code_agent.py", "src/appworld_openai_agents_sdk/prompts")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
