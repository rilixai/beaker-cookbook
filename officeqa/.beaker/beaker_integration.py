"""Beaker integration for the OfficeQA Pro grounded-reasoning agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock

from beaker import (
    Case,
    CaseResult,
    CaseScore,
    Check,
    Integration,
    RepositoryRunSetup,
    RolloutRuntime,
    SetupRuntime,
    repository,
)
from pydantic import BaseModel, Field


HOSTED_CORPUS = "parsed"
_CORPUS_LOCK = Lock()


class Row(BaseModel):
    """One labeled OfficeQA Pro question."""

    uid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class Setup(RepositoryRunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        del runtime
        yield Case(
            id=row.uid,
            input={"uid": row.uid, "question": row.question},
            expected={"answer": row.answer},
        )


def _litellm_model(selected_model: str | None) -> str | None:
    """Translate Beaker's canonical provider:model form to LiteLLM syntax."""

    if selected_model is None:
        return None
    provider, separator, model = selected_model.partition(":")
    return f"{provider}/{model}" if separator else selected_model


def _ensure_hosted_corpus() -> Path:
    """Fetch and verify the lightweight parsed corpus once per evaluator."""

    from officeqa import config
    from officeqa.data.corpus import IncompleteCorpusError, ensure_corpus, fetch_corpus

    with _CORPUS_LOCK:
        try:
            return ensure_corpus(
                HOSTED_CORPUS,
                expected_documents=config.EXPECTED_CORPUS_DOCUMENTS,
            )
        except (FileNotFoundError, IncompleteCorpusError):
            fetch_corpus((HOSTED_CORPUS,))
            return ensure_corpus(
                HOSTED_CORPUS,
                expected_documents=config.EXPECTED_CORPUS_DOCUMENTS,
            )


async def run_case(*, case_input: object, runtime: RolloutRuntime) -> CaseResult:
    from officeqa.config import RunConfig
    from officeqa.data.dataset import EvalRecord
    from officeqa.runner import run_one_async

    if not isinstance(case_input, dict):
        raise TypeError("OfficeQA case input must be an object")
    uid = str(case_input["uid"])
    question = str(case_input["question"])

    # Ground truth stays in the trusted scorer. The application runner only
    # receives the question and neutral corpus manifest through AgentInput.
    sample = EvalRecord(
        uid=uid,
        question=question,
        answer="",
        source_docs="",
        source_files="",
        difficulty="",
    )
    selected_model = _litellm_model(runtime.model)
    cfg = (
        RunConfig(model=selected_model, corpus=HOSTED_CORPUS)
        if selected_model is not None
        else RunConfig(corpus=HOSTED_CORPUS)
    )

    with runtime.trace.stage(
        "officeqa.prepare_corpus",
        inputs={"representation": HOSTED_CORPUS},
    ) as stage:
        corpus_root = await asyncio.to_thread(_ensure_hosted_corpus)
        stage.output({"representation": HOSTED_CORPUS})

    with runtime.trace.stage("officeqa.run_one", inputs={"uid": uid, "question": question}) as stage:
        result = await run_one_async(sample, cfg=cfg, corpus_root=corpus_root)
        output = {
            "final_answer": result.final_answer,
            "status": result.status,
            "error": result.error,
            "steps": result.steps,
            "tool_calls": result.tool_calls,
        }
        stage.output(output)

    return CaseResult(output=output, output_kind="record")


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    from officeqa.evaluation.reward import score_answer

    del case_files_dir
    if not isinstance(case.expected, dict):
        raise TypeError("OfficeQA expected value must be an object")
    if not isinstance(result.output, dict):
        raise TypeError("OfficeQA result output must be an object")

    expected = str(case.expected["answer"])
    raw_prediction = result.output.get("final_answer")
    prediction = raw_prediction if isinstance(raw_prediction, str) else ""
    exact = float(score_answer(expected, prediction, tolerance=0.0)) if prediction.strip() else 0.0
    status = str(result.output.get("status") or "unknown")
    error = result.output.get("error")
    message = None
    if not exact:
        message = f"OfficeQA scorer mismatch; rollout status={status}"
        if error:
            message += f"; error={error}"

    return CaseScore(
        objective=exact,
        field_scores={"correctness_0pct": exact},
        checks=(
            Check(
                name="final_answer",
                verdict="pass" if exact else "fail",
                expected=expected,
                predicted=prediction or None,
                message=message,
            ),
        ),
    )


integration = Integration(
    targets=repository(("src/officeqa/agent",)),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
