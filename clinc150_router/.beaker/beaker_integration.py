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
    repository,
)
from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient

from clinc150_router.agent import question, route
from clinc150_router.contracts import (
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


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    if runtime.model:
        raise ValueError("This recipe uses TypeSafe decisions. Select decision models for comparison.")
    message = Message.model_validate(case_input)
    routing_question = question(Path.cwd() / "prompts")
    async with AsyncTypeSafeClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        prediction = await route(message.text, client=client, routing_question=routing_question, trace=runtime.trace)
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
