"""Optimize scenario goal completion using complete AppWorld train/dev scenarios."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from appworld_setup import PROJECT, prepare_appworld
from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    RepositoryRunSetup,
    RepositoryRunSetupResult,
    inference_target,
    repository,
)
from beaker.tracing.integrations import openai_agents
from pydantic import BaseModel, Field, model_validator

from appworld_openai_agents_sdk.models import ModelProfile


class _GatewayModelProfile(ModelProfile):
    """Leave provider-specific generation settings to the inference gateway."""

    def settings(self) -> dict:
        return {"tool_choice": "auto"}


@asynccontextmanager
async def _model_config(runtime):
    from agents import OpenAIChatCompletionsModel
    from agents.run import RunConfig
    from openai import AsyncOpenAI

    if not runtime.model:
        yield ModelProfile.from_toml(PROJECT / "configs/model.toml"), RunConfig(tracing_disabled=False)
        return

    target = inference_target(runtime)
    async with AsyncOpenAI(api_key=target.api_key, base_url=target.base_url) as client:
        yield (
            _GatewayModelProfile(name=target.model, api_type="chat_completions"),
            RunConfig(
                model=OpenAIChatCompletionsModel(model=target.model, openai_client=client),
                tracing_disabled=False,
            ),
        )


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
    prepare_appworld()
    from agents import set_trace_processors, set_tracing_disabled
    from agents.tracing import get_trace_provider
    from appworld import AppWorld, evaluate_task
    from appworld.apps.lib.models.db import CachedDBHandler

    from appworld_openai_agents_sdk.code_agent import run_code_agent_on_tasks
    from appworld_openai_agents_sdk.runner import MAX_STEPS, PROMPTS_DIR, RANDOM_SEED

    task_ids = [task["task_id"] for task in case_input["tasks"]]
    experiment = f"beaker-{uuid4().hex}"
    # The application disables SDK telemetry by default. Enable it only in this
    # isolated evaluation, and keep traces in Beaker rather than OpenAI export.
    provider = get_trace_provider()
    processors = list(provider._multi_processor._processors)
    disabled = provider._disabled
    try:
        set_trace_processors([])
        set_tracing_disabled(False)
        async with _model_config(runtime) as (profile, run_config):
            with openai_agents.registered(runtime.trace):
                await run_code_agent_on_tasks(
                    experiment_name=experiment,
                    task_ids=task_ids,
                    profile=profile,
                    prompt_file_path=str(PROMPTS_DIR / "react_code_agent/instructions.txt"),
                    appworld_config={"random_seed": RANDOM_SEED},
                    logger_config={"color": False, "verbose": False},
                    max_steps=MAX_STEPS,
                    run_config=run_config,
                    raise_provider_errors=True,
                )
    finally:
        set_trace_processors(processors)
        set_tracing_disabled(disabled)
        AppWorld.close_all()
    results = {}
    for task_id in task_ids:
        try:
            results[task_id] = evaluate_task(task_id, experiment, save_report=False).to_dict()
        finally:
            CachedDBHandler.reset()
    return CaseResult(output={"tasks": results}, output_kind="record")


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
    return CaseScore(
        objective=sgc,
        field_scores={"scenario_goal_completion": sgc, "task_goal_completion": sum(successes) / len(successes)},
        checks=tuple(checks),
    )


integration = Integration(
    targets=repository(("src/appworld_openai_agents_sdk/code_agent.py", "src/appworld_openai_agents_sdk/prompts")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
