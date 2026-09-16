"""Evaluate the real AppWorld agent against official simulator assertions."""

import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from appworld_resources import prepare_resources
from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RepositoryRunSetup,
    RepositoryRunSetupResult,
    RolloutRuntime,
    SetupRuntime,
    repository,
)
from pydantic import BaseModel, Field


class Row(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9]+_[0-9]+$")
    instruction: str = Field(min_length=1)
    requirements: list[str] = Field(min_length=1)
    split: Literal["train", "dev"]
    data_version: Literal["0.2.0"]


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(self, *, runtime: SetupRuntime):
        self.data_root = prepare_resources()
        yield RepositoryRunSetupResult(metadata={"data_version": "0.2.0"})

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        task_dir = self.data_root / "data" / "tasks" / row.id
        specs = json.loads((task_dir / "specs.json").read_text())
        requirements = json.loads((task_dir / "ground_truth" / "test_data.json").read_text())
        split_ids = (self.data_root / "data" / "datasets" / f"{row.split}.txt").read_text().splitlines()
        if row.id not in split_ids or specs["instruction"] != row.instruction:
            raise ValueError(f"Dataset row does not match official AppWorld task {row.id}")
        if row.requirements != [item["requirement"] for item in requirements]:
            raise ValueError(f"Requirements do not match official AppWorld task {row.id}")
        yield Case(
            id=row.id,
            input={"task_id": row.id, "instruction": row.instruction},
            expected={"requirements": row.requirements},
            metadata={"split": row.split, "data_version": row.data_version},
        )


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    from agents import set_trace_processors, set_tracing_disabled
    from agents.run import RunConfig
    from appworld import AppWorld, evaluate_tasks
    from appworld.common.path_store import path_store
    from beaker.tracing.integrations import openai_agents

    from appworld_openai_agents_sdk.code_agent import run_code_agent_on_tasks
    from appworld_openai_agents_sdk.models import ModelProfile
    from appworld_openai_agents_sdk.runner import MAX_STEPS, PROMPTS_DIR, RANDOM_SEED

    if not isinstance(case_input, dict):
        raise TypeError("AppWorld case input must be an object")
    task_id = str(case_input["task_id"])
    data_root = prepare_resources()
    project_root = Path(__file__).resolve().parent.parent
    profile = ModelProfile.from_toml(project_root / "configs" / "model.toml")
    if runtime.model is not None:
        provider, _, model = runtime.model.partition(":")
        if provider != "openai" or not model:
            raise ValueError("This native Responses agent supports openai:<model> overrides only")
        profile = ModelProfile(name=model)
    previous_root = path_store.root
    try:
        with tempfile.TemporaryDirectory(prefix="beaker-appworld-case-") as temp:
            root = Path(temp)
            (root / "data").symlink_to(data_root / "data", target_is_directory=True)
            path_store.update_root(str(root))
            # Each Beaker repository case runs in its own evaluator process.
            # Export its SDK spans through Beaker, without the default OpenAI exporter.
            set_trace_processors([])
            set_tracing_disabled(False)
            with openai_agents.registered(runtime.trace):
                await run_code_agent_on_tasks(
                    experiment_name="beaker",
                    task_ids=[task_id],
                    profile=profile,
                    prompt_file_path=str(PROMPTS_DIR / "react_code_agent" / "instructions.txt"),
                    appworld_config={"random_seed": RANDOM_SEED},
                    logger_config={"color": False, "verbose": False},
                    max_steps=MAX_STEPS,
                    run_config=RunConfig(tracing_disabled=False),
                )
            metrics = evaluate_tasks(
                task_ids=[task_id],
                experiment_name="beaker",
                suppress_errors=True,
                include_details=True,
                save_reports=False,
            )
            tracker = metrics["individual"][task_id]
            return CaseResult(output={"task_id": task_id, "evaluation": tracker}, output_kind="record")
    finally:
        AppWorld.close_all()
        set_tracing_disabled(True)
        path_store.update_root(previous_root)


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    if not isinstance(result.output, dict) or not isinstance(case.expected, dict):
        raise ValueError("AppWorld scoring requires the evaluator result and task requirements")
    evaluation = result.output.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("AppWorld evaluator result is missing")
    checks = []
    for key, verdict in (("passes", "pass"), ("failures", "fail")):
        for assertion in evaluation[key]:
            checks.append(
                Check(
                    name=assertion["requirement"],
                    verdict=verdict,
                    message=assertion.get("trace") or None,
                )
            )
    expected = case.expected["requirements"]
    if sorted(check.name for check in checks) != sorted(expected):
        raise ValueError("AppWorld evaluator did not report every expected assertion")
    tgc = float(all(check.verdict == "pass" for check in checks))
    if bool(evaluation["success"]) != bool(tgc):
        raise ValueError("AppWorld success flag disagrees with assertion results")
    return CaseScore(objective=tgc, field_scores={"task_goal_completion": tgc}, checks=tuple(checks))


integration = Integration(
    targets=repository(
        (
            "src/appworld_openai_agents_sdk/code_agent.py",
            "src/appworld_openai_agents_sdk/models.py",
            "src/appworld_openai_agents_sdk/prompts",
        )
    ),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
