"""Optimize the CLINC150 router against fixed labels, data and scoring."""

from __future__ import annotations

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
)
from openai import AsyncOpenAI
from pydantic import ValidationError

from clinc150_router.agent import ChatModel, route, system_prompt
from clinc150_router.contracts import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    LABELS,
    OUT_OF_SCOPE,
    REQUEST_TIMEOUT_SECONDS,
    Prediction,
)
from clinc150_router.data import UPSTREAM, Expected, Message, Row


# Weight by inverse frequency on the official optimization validation split.
# CaseScore caps each case at 1, so normalize to the rarer class's weight.
_, _, VALIDATION_ROWS_PER_INTENT, VALIDATION_OOS_ROWS = UPSTREAM["validation"]
IN_SCOPE_WEIGHT = VALIDATION_OOS_ROWS / ((len(LABELS) - 1) * VALIDATION_ROWS_PER_INTENT)


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected.model_dump())


def _client_for(runtime: RolloutRuntime[Any]) -> ChatModel:
    """The model the run selected, through Beaker's inference gateway; else the pinned default.

    A run started with a model choice (model-swap runs) resolves to a gateway
    endpoint, a run-scoped token and a canonical ``<provider>:<model>`` name;
    the gateway applies the run's reasoning effort, so none is sent. With no
    selection the pinned model and ``OPENAI_API_KEY`` are used as-is, which
    inside a hosted rollout is Beaker's provider proxy.
    """
    if runtime.model:
        target = inference_target(runtime)
        client = AsyncOpenAI(api_key=target.api_key, base_url=target.base_url, timeout=REQUEST_TIMEOUT_SECONDS)
        return ChatModel(client=client, provider=target.model.partition(":")[0], model=target.model)
    return ChatModel(
        client=AsyncOpenAI(timeout=REQUEST_TIMEOUT_SECONDS),
        provider="openai",
        model=DEFAULT_MODEL,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    message = Message.model_validate(case_input)
    chat = _client_for(runtime)
    try:
        # Repository candidates execute with source_dir as their working directory.
        # There is deliberately no fallback to seed prompts outside that candidate.
        prompt = system_prompt(Path.cwd() / "prompts")
        prediction = await route(message.text, chat=chat, prompt=prompt, trace=runtime.trace)
    finally:
        await chat.client.close()
    return CaseResult(output=prediction.model_dump(mode="json"), output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    expected = Expected.model_validate(case.expected)
    try:
        prediction = Prediction.model_validate(result.output)
        intent, error = prediction.intent, prediction.error
    except ValidationError:
        intent, error = None, "Invalid prediction record."
    correct = float(error is None and intent == expected.intent)
    is_oos = expected.intent == OUT_OF_SCOPE
    return CaseScore(
        objective=correct * (1.0 if is_oos else IN_SCOPE_WEIGHT),
        field_scores={"accuracy": correct, "oos_recall" if is_oos else "in_scope_accuracy": correct},
        checks=(
            Check(
                name="intent",
                verdict="pass" if correct else "fail",
                expected=expected.intent,
                predicted=intent,
                message=error,
            ),
        ),
    )


integration = Integration(
    targets=repository(paths=("prompts", "src/clinc150_router/agent.py")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
