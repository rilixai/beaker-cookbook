"""Optimize the LiveNewsBench web-search agent; the model, web rules and grader stay fixed.

`integration` runs an open model on the You.com router with You.com search.
`openai_integration` runs GPT with OpenAI's built-in web search, as a baseline.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    JsonValue,
    RetryableCaseError,
    RolloutRuntime,
    RunSetup,
    SetupRuntime,
    inference_target,
    repository,
    scoring_inference_target,
)
from openai import AsyncOpenAI

from livenewsbench import openai_search, youcom
from livenewsbench.chat import ChatModel, TransientServiceError
from livenewsbench.data import Expected, Question, Row
from livenewsbench.grading import GRADER_MODEL, grade
from livenewsbench.runner import (
    MAX_MODEL_CALLS,
    MODEL_ENV,
    load_agent,
    load_openai_agent,
    run_openai_rollout,
    run_rollout,
)
from livenewsbench.tools import WebTools


# Repository candidates run with source_dir as the working directory. Load the
# candidate's agent from there, never the seed installed in the environment.
AGENT_DIR = Path("src/livenewsbench/agent")
OPENAI_AGENT_DIR = Path("src/livenewsbench/openai_agent")


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input=row.input.model_dump(),
            expected=row.expected.model_dump(),
            metadata=row.metadata.model_dump(),
        )


def _chat_model(runtime: RolloutRuntime[Any], key: str) -> ChatModel:
    if runtime.model:
        # An optimization model picked in Beaker goes through Beaker's gateway, for comparison runs.
        target = inference_target(runtime)
        return ChatModel(
            client=AsyncOpenAI(base_url=target.base_url, api_key=target.api_key, max_retries=5, timeout=300),
            model=target.model,
            provider=target.model.split(":", 1)[0] if ":" in target.model else "openai",
            server_address=urlsplit(target.base_url).hostname or "",
            max_calls=MAX_MODEL_CALLS,
            trace=runtime.trace,
        )
    return ChatModel(
        client=youcom.router_client(key),
        model=os.environ.get(MODEL_ENV) or youcom.DEFAULT_MODEL,
        provider="youcom",
        server_address=youcom.ROUTER_HOST,
        max_calls=MAX_MODEL_CALLS,
        trace=runtime.trace,
        request_options=youcom.REQUEST_OPTIONS,
    )


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    question = Question.model_validate(case_input)
    agent = load_agent(Path.cwd() / AGENT_DIR)
    key = youcom.api_key()
    chat = _chat_model(runtime, key)
    async with chat.client, youcom.YouComWeb(key) as backend:
        web = WebTools(backend=backend, trace=runtime.trace)
        try:
            rollout = await run_rollout(question.question, agent=agent, chat=chat, web=web)
        except TransientServiceError as exc:
            raise RetryableCaseError(str(exc)) from exc
    runtime.trace.evidence("usage", rollout.usage())
    return CaseResult(output=rollout.answer, output_kind="value")


async def run_openai_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    if runtime.model:
        # Beaker's gateway accepts built-in web search only from its sandbox provider proxy.
        raise RuntimeError(
            "livenewsbench-openai calls OpenAI directly for its built-in web search. Launch it without "
            f"--optimization-model, and set {MODEL_ENV} on the agent to change the model."
        )
    question = Question.model_validate(case_input)
    agent = load_openai_agent(Path.cwd() / OPENAI_AGENT_DIR)
    model = openai_search.ResponsesModel(
        client=openai_search.openai_client(),
        model=os.environ.get(MODEL_ENV) or openai_search.DEFAULT_MODEL,
        max_calls=MAX_MODEL_CALLS,
        trace=runtime.trace,
    )
    async with model.client:
        try:
            rollout = await run_openai_rollout(question.question, agent=agent, model=model)
        except TransientServiceError as exc:
            raise RetryableCaseError(str(exc)) from exc
    runtime.trace.evidence("usage", rollout.usage())
    return CaseResult(output=rollout.answer, output_kind="value")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    question = Question.model_validate(case.input)
    gold = Expected.model_validate(case.expected).answer
    predicted = result.output if isinstance(result.output, str) else ""
    target = scoring_inference_target()
    if target is None:
        client, model = AsyncOpenAI(), GRADER_MODEL
    else:
        client, model = AsyncOpenAI(base_url=target.base_url, api_key=target.api_key), target.model
    async with client:
        verdict = await grade(client, model=model, question=question.question, gold=gold, predicted=predicted)
    correct = verdict == "CORRECT"
    return CaseScore(
        objective=float(correct),
        field_scores={"accuracy": float(correct)},
        checks=(
            Check(
                name="answer",
                verdict="pass" if correct else "fail",
                expected=gold,
                predicted=predicted,
                message=verdict,
            ),
        ),
    )


integration = Integration(
    targets=repository(paths=(str(AGENT_DIR),)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)

openai_integration = Integration(
    targets=repository(paths=(str(OPENAI_AGENT_DIR),)),
    run_setup=Setup,
    run_case=run_openai_case,
    score_case=score_case,
)
