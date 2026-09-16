"""One complete AppWorld scenario per case; maximize Scenario Goal Completion."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from beaker import (
    Case,
    CaseFile,
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
from beaker.tracing.integrations import openai_agents
from pydantic import BaseModel, Field


class TaskInput(BaseModel):
    task_id: str = Field(pattern=r"^[a-zA-Z0-9]+_[0-9]+$")
    instruction: str = Field(min_length=1)


class Row(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9]+$")
    tasks: list[TaskInput] = Field(min_length=1)
    requirements: dict[str, list[str]]
    split: Literal["train", "dev"]


def ensure_appworld_installed() -> None:
    import importlib.util

    spec = importlib.util.find_spec("appworld")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the project's AppWorld dependency first.")
    if not (Path(spec.origin).parent / "apps/admin/models.py").is_file():
        subprocess.run([sys.executable, "-m", "appworld.cli", "install"], check=True)


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    @asynccontextmanager
    async def prepare_run(self, *, runtime: SetupRuntime):
        del runtime
        ensure_appworld_installed()
        local_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="beaker-appworld-data-") as temp:
            self.data = local_root / "data"
            if not (self.data / "datasets/train.txt").is_file():
                subprocess.run(
                    [sys.executable, "-m", "appworld.cli", "download", "data"],
                    cwd=temp,
                    env={**os.environ, "APPWORLD_ROOT": temp},
                    check=True,
                )
                self.data = Path(temp) / "data"
            yield RepositoryRunSetupResult()

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        ids = [task.task_id for task in row.tasks]
        split_ids = (self.data / "datasets" / f"{row.split}.txt").read_text().splitlines()
        scenario_ids = [task_id for task_id in split_ids if task_id.rsplit("_", 1)[0] == row.id]
        if set(ids) != set(scenario_ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{row.id}: expected every task in the source scenario exactly once")
        if set(row.requirements) != set(ids):
            raise ValueError(f"{row.id}: expected requirements for every task")
        for task in row.tasks:
            specs = json.loads((self.data / "tasks" / task.task_id / "specs.json").read_text())
            if specs["instruction"] != task.instruction:
                raise ValueError(f"Task instruction differs from source: {task.task_id}")
            labels = json.loads((self.data / "tasks" / task.task_id / "ground_truth/test_data.json").read_text())
            if row.requirements[task.task_id] != [label["requirement"] for label in labels]:
                raise ValueError(f"Task requirements differ from source: {task.task_id}")
        archive = runtime.case_files_dir(row.id) / "world.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for name in ("api_docs", "base_dbs", "version.txt"):
                bundle.add(self.data / name, arcname=f"data/{name}")
            for task_id in ids:
                bundle.add(self.data / "tasks" / task_id, arcname=f"data/tasks/{task_id}")
        yield Case(
            id=row.id,
            input={"tasks": [task.model_dump() for task in row.tasks]},
            expected={"requirements": row.requirements},
            files=(CaseFile(name="world.tar.gz", path=archive),),
            metadata={"source_split": row.split, "scenario_id": row.id},
        )


# AppWorld maintains process-wide paths and simulator state. Serialize cases.
_case_lock = asyncio.Lock()


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    ensure_appworld_installed()
    from agents import set_tracing_disabled
    from agents.run import RunConfig
    from agents.tracing import get_trace_provider
    from appworld.common.path_store import path_store
    from appworld.evaluator import evaluate_task

    from appworld_openai_agents_sdk.code_agent import run_code_agent_on_tasks
    from appworld_openai_agents_sdk.models import ModelProfile
    from appworld_openai_agents_sdk.runner import MAX_STEPS, PROMPTS_DIR, RANDOM_SEED

    if not isinstance(case_input, dict) or not isinstance(case_input.get("tasks"), list):
        raise ValueError("Expected scenario tasks")
    tasks = [TaskInput.model_validate(task) for task in case_input["tasks"]]
    profile = ModelProfile(name="gpt-5.6-sol")
    if runtime.model:
        provider, _, name = runtime.model.partition(":")
        if provider != "openai" or not name:
            raise ValueError("This application supports OpenAI models only")
        profile = ModelProfile(name=name)
    async with _case_lock:
        with tempfile.TemporaryDirectory(prefix="beaker-appworld-case-") as temp:
            with tarfile.open(runtime.case_files_dir / "world.tar.gz") as bundle:
                bundle.extractall(temp, filter="data")
            previous_root = path_store.root
            previous_disabled = get_trace_provider()._disabled
            path_store.update_root(temp)
            try:
                # Enable telemetry only for this evaluation scope and restore it below.
                if runtime.trace.enabled:
                    set_tracing_disabled(False)
                with openai_agents.registered(runtime.trace):
                    await run_code_agent_on_tasks(
                        experiment_name="beaker",
                        task_ids=[task.task_id for task in tasks],
                        profile=profile,
                        prompt_file_path=str(PROMPTS_DIR / "react_code_agent/instructions.txt"),
                        appworld_config={"random_seed": RANDOM_SEED},
                        logger_config={"color": False, "verbose": False},
                        max_steps=MAX_STEPS,
                        run_config=RunConfig(tracing_disabled=not runtime.trace.enabled),
                        raise_provider_errors=True,
                    )
                # The benchmark evaluator runs outside candidate tracing.
                outcomes = {
                    task.task_id: evaluate_task(task.task_id, experiment_name="beaker", save_report=False).to_dict()
                    for task in tasks
                }
                return CaseResult(output={"tasks": outcomes}, output_kind="record")
            finally:
                set_tracing_disabled(previous_disabled)
                path_store.update_root(previous_root)


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    if not isinstance(result.output, dict) or not isinstance(result.output.get("tasks"), dict):
        raise ValueError("Missing AppWorld evaluation results")
    outcomes = result.output["tasks"]
    expected_ids = set(case.expected["requirements"])
    if set(outcomes) != expected_ids:
        raise ValueError("AppWorld results do not cover the complete scenario")
    checks = []
    successes = []
    for task_id, outcome in outcomes.items():
        successes.append(bool(outcome["success"]))
        for verdict, field in (("pass", "passes"), ("fail", "failures")):
            for criterion in outcome[field]:
                checks.append(
                    Check(
                        name=criterion["requirement"],
                        group=task_id,
                        verdict=verdict,
                        message=criterion.get("trace") or None,
                    )
                )
    sgc = float(all(successes))
    return CaseScore(
        objective=sgc,
        field_scores={"scenario_goal_completion": sgc, "task_goal_completion": sum(successes) / len(successes)},
        checks=tuple(checks),
    )


integration = Integration(
    targets=repository(("src/appworld_openai_agents_sdk", "configs")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
