"""Optimize an MMDM decision agent; keep data, the request guard, decision calls and scoring fixed.

Editable: ``prompts/`` and ``src/mmdm/agent.py``. Fixed: the dataset, ``decide.py``
(the request guard and every traced model call), ``scoring.py`` and the
unmodified upstream ``mmdm_eval`` toolkit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
from pydantic import ConfigDict, Field

from mmdm import agent
from mmdm.data import Expected, Question, Row
from mmdm.decide import (
    CHAT_TIMEOUT_SECONDS,
    DEFAULT_TARGET,
    ChatModel,
    DecisionTarget,
    default_chat_model,
    run_question,
)
from mmdm.scoring import score


class CaseInput(Question):
    """The question plus its case ID, which names the trace stage. Never labels or category."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)

    def question_only(self) -> Question:
        return Question.model_validate(self.model_dump(exclude={"id"}))


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.id,
            input={"id": row.id, **row.input.model_dump(mode="json")},
            expected=row.expected.model_dump(mode="json"),
        )


def _decision_target(runtime: RolloutRuntime[Any]) -> DecisionTarget:
    """The run's selected decision model through Beaker's gateway; else native Jev."""
    if runtime.model:
        target = inference_target(runtime)
        return DecisionTarget(
            model=target.model,
            provider=target.model.partition(":")[0] or DEFAULT_TARGET.provider,
            server_address=urlparse(target.base_url).hostname or target.base_url,
            base_url=target.base_url,
            api_key=target.api_key,
        )
    return DEFAULT_TARGET


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
    case = CaseInput.model_validate(case_input)
    question = case.question_only()
    # Repository candidates execute with source_dir as their working directory.
    instructions = agent.load_instructions(Path.cwd() / "prompts")
    with runtime.trace.stage(
        "mmdm.question", inputs={"id": case.id, "task": question.task, "option_count": len(question.options)}
    ) as stage:
        outcome = await run_question(
            question,
            solver=agent.answer,
            instructions=instructions,
            target=_decision_target(runtime),
            trace=runtime.trace,
            chat_factory=lambda: _client_for(runtime),
        )
        output = outcome.record()
        stage.output({k: v for k, v in output.items() if k != "probabilities"})
    if outcome.retryable:
        raise RetryableCaseError(outcome.error or "transient decision API error")
    return CaseResult(output=output, output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    question = CaseInput.model_validate(case.input).question_only()
    scored = score(case.id, question, Expected.model_validate(case.expected), result.output)
    verdict: float | str
    if scored.kind == "soft" and scored.valid:
        verdict = scored.objective
    else:
        verdict = "pass" if scored.objective == 1.0 else "fail"
    return CaseScore(
        objective=scored.objective,
        field_scores=scored.field_scores,
        checks=(
            Check(
                name="answer",
                verdict=verdict,  # type: ignore[arg-type]
                expected=list[JsonValue](scored.expected_top),
                predicted=None if scored.predicted_top is None else list[JsonValue](scored.predicted_top),
                message=scored.error,
            ),
        ),
    )


integration = Integration(
    targets=repository(paths=("prompts", "src/mmdm/agent.py")),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
