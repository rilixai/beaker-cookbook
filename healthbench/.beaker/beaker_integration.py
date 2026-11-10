"""Optimize the HealthBench agent's system prompt, graded by the paper's per-criterion rubric judge."""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
    scoring_inference_target,
)
from openai import AsyncOpenAI
from pydantic import BaseModel

from healthbench.agent.agent import HealthBenchAgent
from healthbench.config import DEFAULT_GRADER_MODEL, HealthBenchConfig
from healthbench.data.dataset import HealthBenchRecord, RubricItem
from healthbench.evaluation.grader import RubricGrader
from healthbench.evaluation.scoring import OVERALL, case_metrics


AGENT_MODEL = "gpt-6-luna"
PROMPTS_FILE = Path("src/healthbench/agent/prompts.py")
AXIS_TAG_PREFIX = "axis:"


class Message(BaseModel):
    role: str
    content: str


class Rubric(BaseModel):
    criterion: str
    points: float
    tags: list[str] = []


class Row(BaseModel):
    prompt_id: str
    prompt: list[Message]
    rubrics: list[Rubric]
    example_tags: list[str] = []


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.prompt_id,
            input={"prompt": [m.model_dump() for m in row.prompt]},
            expected={
                "rubrics": [r.model_dump() for r in row.rubrics],
                "example_tags": row.example_tags,
            },
        )


def _candidate_system_prompt() -> str:
    # Repository candidates run with source_dir as the working directory; read the candidate's prompt, not the seed.
    spec = importlib.util.spec_from_file_location("candidate_healthbench_prompts", Path.cwd() / PROMPTS_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEM_PROMPT


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    config = HealthBenchConfig()
    prompt = [Message.model_validate(m).model_dump() for m in case_input["prompt"]]  # type: ignore[index]
    if runtime.model:
        target = inference_target(runtime)
        client = AsyncOpenAI(base_url=target.base_url, api_key=target.api_key)
        model = target.model
    else:
        # Hosted runs route this through Beaker's provider proxy; locally it uses OPENAI_API_KEY.
        client = AsyncOpenAI()
        model = AGENT_MODEL
    agent = HealthBenchAgent(
        client,
        model=model,
        temperature=config.task_temperature,
        max_tokens=config.task_max_tokens,
        system_prompt=_candidate_system_prompt(),
    )
    record = HealthBenchRecord(prompt_id="", prompt=prompt, rubrics=[])
    with runtime.trace.model_call(
        operation="chat",
        provider="openai",
        model=model,
        server_address="api.openai.com",
        input_messages=prompt,
    ) as call:
        async with client:
            output = await agent.respond(record)
        call.output(output.response)
        call.usage(
            input_tokens=output.usage.get("prompt_tokens", 0),
            output_tokens=output.usage.get("completion_tokens", 0),
        )
    return CaseResult(output={"response": output.response}, output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    expected: dict[str, Any] = case.expected  # type: ignore[assignment]
    rubrics = [RubricItem(r["criterion"], r["points"], list(r["tags"])) for r in expected["rubrics"]]
    prompt = [Message.model_validate(m).model_dump() for m in case.input["prompt"]]  # type: ignore[index]
    response: str = result.output["response"]  # type: ignore[index]

    target = scoring_inference_target()
    if target is None:
        client, judge_model = AsyncOpenAI(), DEFAULT_GRADER_MODEL
    else:
        client, judge_model = AsyncOpenAI(base_url=target.base_url, api_key=target.api_key), target.model
    async with client:
        grader = RubricGrader(client, model=judge_model, max_retries=HealthBenchConfig().grader_max_retries)
        grades = await grader.grade(prompt, response, rubrics)

    metrics = case_metrics(grades, expected["example_tags"])
    # Beaker scores live in [0, 1]; the paper clips the mean, we clip each case.
    field_scores = {
        name: min(1.0, max(0.0, value))
        for name, value in metrics.items()
        if name == OVERALL or name.startswith(AXIS_TAG_PREFIX)
    }
    checks = tuple(
        Check(
            name=grade.criterion,
            # A met negative-point criterion is a harm, so it fails.
            verdict="pass" if grade.criteria_met == (grade.points > 0) else "fail",
            message=grade.explanation,
            description=f"{grade.points:+g} points",
        )
        for grade in grades
    )
    return CaseScore(objective=field_scores[OVERALL], field_scores=field_scores, checks=checks)


integration = Integration(
    targets=repository(paths=(str(PROMPTS_FILE),)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
