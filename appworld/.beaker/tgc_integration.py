"""Binary Task Goal Completion: one task per case, no partial credit."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from appworld_setup import prepare_appworld
from beaker import Case, CaseScore, Integration, RepositoryRunSetup, RepositoryRunSetupResult
from beaker_integration import TaskInput
from beaker_integration import integration as scenario_integration
from beaker_integration import run_case as run_tasks
from beaker_integration import score_case as score_tasks
from pydantic import BaseModel


class Row(BaseModel):
    id: str
    input: TaskInput
    expected: list[dict]


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(self, *, runtime):
        self.root = prepare_appworld()
        yield RepositoryRunSetupResult()

    async def load_cases(self, row: Row, *, runtime) -> AsyncIterator[Case]:
        import json

        allowed = {
            task_id
            for split in ("train", "dev")
            for task_id in (self.root / f"data/datasets/{split}.txt").read_text().splitlines()
        }
        if row.id != row.input.task_id or row.id not in allowed:
            raise ValueError("Each case must identify one official train/dev task")
        task_dir = self.root / "data/tasks" / row.id
        specs = json.loads((task_dir / "specs.json").read_text())
        requirements = json.loads((task_dir / "ground_truth/test_data.json").read_text())
        if row.input.instruction != specs["instruction"] or row.expected != requirements:
            raise ValueError(f"Task {row.id} does not match the official instructions and requirements")
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected)


async def run_case(*, case_input, runtime):
    return await run_tasks(case_input={"tasks": [case_input]}, runtime=runtime)


async def score_case(*, case, result, case_files_dir):
    grouped_case = Case(id=case.id, input={"tasks": [case.input]}, expected={case.id: case.expected})
    score = await score_tasks(case=grouped_case, result=result, case_files_dir=case_files_dir)
    tgc = score.field_scores["task_goal_completion"]
    return CaseScore(objective=tgc, field_scores={"task_goal_completion": tgc}, checks=score.checks)


integration = Integration(
    targets=scenario_integration.targets,
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
