"""Optimize routing descriptions while keeping labels and exact-match scoring fixed."""

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
    repository,
)
from typesafe_sdk import AsyncTypeSafeClient

from jev_banking77.agent import MODEL, Prediction, classify, question
from jev_banking77.data import Expected, Message, Row


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(id=row.id, input=row.input.model_dump(), expected=row.expected.model_dump())


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    if runtime.model:
        raise ValueError("This recipe pins native Jev; do not pass a chat optimization-model override.")
    message = Message.model_validate(case_input)
    # Repository candidates execute with source_dir as their working directory.
    # There is deliberately no fallback to seed prompts outside that candidate.
    routing_question = question(Path.cwd() / "prompts")
    with runtime.trace.model_call(
        operation="systemone",
        provider="typesafe",
        model=MODEL,
        server_address="api.typesafe.ai",
        input_messages={"state": message.text, "questions": {"intent": routing_question.model_dump(mode="json")}},
    ) as call:
        async with AsyncTypeSafeClient(timeout=60) as client:
            prediction = await classify(message.text, client=client, routing_question=routing_question)
        call.output(prediction.model_dump(mode="json"))
        call.usage(input_tokens=prediction.input_tokens, output_tokens=prediction.output_tokens)
    return CaseResult(output=prediction.model_dump(mode="json"), output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    expected = Expected.model_validate(case.expected)
    prediction = Prediction.model_validate(result.output)
    correct = prediction.intent == expected.intent
    return CaseScore(
        objective=float(correct),
        field_scores={"accuracy": float(correct)},
        checks=(
            Check(
                name="intent",
                verdict="pass" if correct else "fail",
                expected=expected.intent,
                predicted=prediction.intent,
            ),
        ),
    )


integration = Integration(
    targets=repository(paths=("prompts",)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
