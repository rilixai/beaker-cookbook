"""Optimize the CLINC150 router while keeping labels, data and exact-match scoring fixed."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Final

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

from clinc150_router.agent import OUT_OF_SCOPE, ChatModel, Prediction, route
from clinc150_router.data import Expected, Message, Row


# Pinned here, not taken from the app, so editing the app never changes the
# model a run without a model choice evaluates. Low effort: routing is
# latency-sensitive.
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT: Final = "low"
REQUEST_TIMEOUT_SECONDS = 60.0


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
        prediction = await route(message.text, chat=chat, prompts_dir=Path.cwd() / "prompts", trace=runtime.trace)
    finally:
        await chat.client.close()
    return CaseResult(output=prediction.model_dump(mode="json"), output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    expected = Expected.model_validate(case.expected)
    prediction = Prediction.model_validate(result.output)
    correct = float(prediction.intent == expected.intent)
    field_scores = {"accuracy": correct}
    if expected.intent == OUT_OF_SCOPE:
        field_scores["oos_recall"] = correct
    else:
        field_scores["in_scope_accuracy"] = correct
    return CaseScore(
        objective=correct,
        field_scores=field_scores,
        checks=(
            Check(
                name="intent",
                verdict="pass" if correct else "fail",
                expected=expected.intent,
                predicted=prediction.intent,
                message=prediction.error,
            ),
        ),
    )


integration = Integration(
    targets=repository(paths=("prompts", "src/clinc150_router/agent.py")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
