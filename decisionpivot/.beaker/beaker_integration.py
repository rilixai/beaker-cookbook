"""Optimize a DecisionPivot decision agent; keep data, the request guard, decision calls and scoring fixed.

Editable: ``prompts/`` and ``src/decisionpivot/agent.py``. Fixed: the data,
``decide.py`` (the request guard and every traced model call), ``scoring.py``
and the pinned upstream generators and scorer.

One case is one contrast group. The agent runs once per item, alone: it never
sees the other members of the group, the answer, the level or the generator's
specification.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

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
)
from openai import AsyncOpenAI
from pydantic import Field

from decisionpivot import agent
from decisionpivot.data import CaseInput, Row
from decisionpivot.decide import (
    CHAT_TIMEOUT_SECONDS,
    DEFAULT_TARGET,
    ChatModel,
    default_chat_model,
    run_item,
)
from decisionpivot.scoring import score_group


class GroupInput(CaseInput):
    """The group's items plus its ID, which names the trace stage. Never gold or level."""

    group_id: str = Field(min_length=1)


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input={"group_id": row.group, **row.input.model_dump(mode="json")},
            expected=row.expected.model_dump(mode="json"),
            metadata={"family": row.input.family, "level": row.expected.level, "layer": row.expected.layer},
        )


# Decision calls always go out as native TypeSafe calls. In a decision-model run
# Beaker's gateway swaps them to the selected model; in a chat-model run they stay
# on Jev. A swapped reply names the target model, so the Jev version pin that local
# runs use is dropped here.
DECISION_TARGET = replace(DEFAULT_TARGET, expected_model=None)


def _client_for(runtime: RolloutRuntime[Any]) -> ChatModel:
    """For chat calls a candidate adds: the run's selected model through the gateway; else the pinned default.

    Copied from clinc150_router. A model choice resolves to a gateway endpoint,
    a run-scoped token and a canonical ``<provider>:<model>`` name; the gateway
    applies the run's reasoning effort, so none is sent.
    """
    if runtime.model:
        target = inference_target(runtime)
        client = AsyncOpenAI(api_key=target.api_key, base_url=target.base_url, timeout=CHAT_TIMEOUT_SECONDS)
        return ChatModel(client=client, provider=target.model.partition(":")[0], model=target.model)
    return default_chat_model()


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime[Any]) -> CaseResult:
    case = GroupInput.model_validate(case_input)
    # Repository candidates execute with source_dir as their working directory.
    instructions = agent.load_instructions(Path.cwd() / "prompts")
    records: list[JsonValue] = []
    with runtime.trace.stage(
        "decisionpivot.group", inputs={"group_id": case.group_id, "family": case.family, "size": len(case.items)}
    ) as group:
        for item in case.items:
            with runtime.trace.stage("decisionpivot.item", inputs={"id": item.id}) as stage:
                outcome = await run_item(
                    item,
                    solver=agent.answer,
                    instructions=instructions,
                    target=DECISION_TARGET,
                    trace=runtime.trace,
                    chat_factory=lambda: _client_for(runtime),
                )
                record = outcome.record()
                stage.output(record)
            if outcome.retryable:
                raise RetryableCaseError(outcome.error or "transient decision API error")
            records.append(record)
        group.output({"failed_items": sum(isinstance(r, dict) and r["error"] is not None for r in records)})
    return CaseResult(output={"items": records}, output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    group = GroupInput.model_validate(case.input)
    row = Row.model_validate(
        {"id": case.id, "input": group.model_dump(exclude={"group_id"}), "expected": case.expected}
    )
    scored = score_group(row, result.output)
    return CaseScore(
        objective=scored.objective,
        field_scores=scored.field_scores,
        checks=tuple(
            Check(
                name=item.id,
                verdict="pass" if item.correct else "fail",
                expected=item.expected,
                predicted=item.predicted,
                message=item.error,
            )
            for item in scored.items
        ),
    )


integration = Integration(
    targets=repository(paths=("prompts", "src/decisionpivot/agent.py")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
