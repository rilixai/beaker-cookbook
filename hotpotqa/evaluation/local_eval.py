"""Local batch evaluation for the HotpotQA agent.

Runs the :class:`~hotpotqa.agent.agent.HotpotQAPydanticAgent` over a set of
:class:`~hotpotqa.data.dataset.HotpotQARecord` cases and scores each answer +
retrieval trace with :mod:`hotpotqa.evaluation.scoring`, aggregating to the
HotpotQA metrics (exact match, answer F1, supporting-title recall).

Cases run concurrently (bounded by ``max_concurrency``). One case failing
never aborts the batch: an errored case counts as ``0`` on the objective and
on every field (a real failure must deflate, never inflate the metrics); a
case with no supervision at all is *unscoreable* and excluded from the
aggregates entirely.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..agent.agent import HotpotQAPydanticAgent
from ..agent.retrieval import build_retrieve_k_fn_for_case
from ..agent.types import HotpotQAAgentOutput
from ..config import HotpotQAConfig
from ..data.dataset import HotpotQARecord
from .scoring import HOTPOTQA_FIELD_WEIGHTS, objective_score, score_prediction


logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Aggregate of the agent's scored performance over a set of records."""

    num_cases: int
    objective: float
    field_accuracies: dict[str, float]
    field_sample_counts: dict[str, int]
    per_case: list[dict[str, Any]] = field(default_factory=list)
    num_scored: int = 0
    num_errored: int = 0
    num_unscoreable: int = 0


async def run_agent_on_record(
    *,
    agent: HotpotQAPydanticAgent,
    record: HotpotQARecord,
    config: HotpotQAConfig,
) -> HotpotQAAgentOutput:
    """Run the agent on one record with the config's retrieval corpus.

    The retriever is built per case so the agent sees the corpus
    ``config.retrieval_mode`` selects (the case's distractor paragraphs, or
    the fullwiki bm25s index) without the agent itself knowing about modes.
    """
    return await agent.forward(
        record=record,
        retrieve_k_fn=build_retrieve_k_fn_for_case(record=record, cfg=config),
    )


async def evaluate_record(
    *,
    record: HotpotQARecord,
    agent: HotpotQAPydanticAgent,
    config: HotpotQAConfig,
    field_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run the agent on ``record`` and score it, returning a per-case result.

    The returned dict always carries ``case_id`` + ``question_type`` and a
    ``kind`` in ``{"scored", "unscoreable"}``. A ``scored`` result also carries
    the ``objective``, the per-field ``field_scores``, and the agent's answer +
    retrieved titles.
    """
    output = await run_agent_on_record(agent=agent, record=record, config=config)
    retrieved_titles = [p.title for p in output.retrieved_paragraphs]
    field_scores = score_prediction(record=record, answer=output.answer, retrieved_titles=retrieved_titles)
    base: dict[str, Any] = {
        "case_id": record.case_id,
        "question_type": record.question_type,
        "level": record.level,
        "question": record.question,
        "answer": output.answer,
        "gold_answer": record.answer,
        "retrieved_titles": retrieved_titles,
        "num_tool_calls": len(output.tool_calls),
    }
    if not field_scores:
        # Nothing to measure (no gold answer, no gold titles): omit from the
        # aggregates rather than scoring it a zero.
        return {**base, "kind": "unscoreable", "objective": 0.0, "field_scores": {}}
    return {
        **base,
        "kind": "scored",
        "objective": objective_score(field_scores, field_weights or HOTPOTQA_FIELD_WEIGHTS),
        "field_scores": field_scores,
    }


async def evaluate_agent_on_records(
    *,
    agent: HotpotQAPydanticAgent,
    records: Sequence[HotpotQARecord],
    config: HotpotQAConfig | None = None,
    field_weights: Mapping[str, float] | None = None,
    max_concurrency: int = 4,
) -> EvalReport:
    """Evaluate ``agent`` over ``records`` and aggregate the HotpotQA metrics."""
    cfg = config or HotpotQAConfig()
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(record: HotpotQARecord) -> dict[str, Any]:
        async with semaphore:
            try:
                return await evaluate_record(
                    record=record,
                    agent=agent,
                    config=cfg,
                    field_weights=field_weights,
                )
            except Exception as exc:  # noqa: BLE001 - one bad case must not abort the batch
                logger.exception("evaluate_record raised for case %s", record.case_id)
                return {
                    "case_id": record.case_id,
                    "question_type": record.question_type,
                    "kind": "error",
                    "objective": 0.0,
                    "field_scores": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }

    per_case = list(await asyncio.gather(*[_run_one(record) for record in records]))
    return _aggregate(per_case)


def _aggregate(per_case: list[dict[str, Any]]) -> EvalReport:
    """Aggregate per-case results into an :class:`EvalReport`.

    Denominator = scored + errored cases. An errored case contributes ``0`` to
    the objective and to every field any case scored, so a failure deflates
    both. An unscoreable case is dropped from every mean — it is not
    measurable.
    """
    # First pass: the union of the fields any scored case produced. Errored
    # cases are then scored 0 on each of them.
    field_names: set[str] = set()
    for result in per_case:
        if result.get("kind") == "scored":
            field_names.update(_field_scores(result))

    field_totals: dict[str, float] = dict.fromkeys(field_names, 0.0)
    field_counts: dict[str, int] = dict.fromkeys(field_names, 0)
    objective_total = 0.0
    denominator = 0
    num_scored = 0
    num_errored = 0
    num_unscoreable = 0

    for result in per_case:
        kind = result.get("kind")
        if kind == "unscoreable":
            num_unscoreable += 1
            continue
        if kind == "error":
            num_errored += 1
            denominator += 1  # denominator grows, totals stay 0
            for name in field_names:
                field_counts[name] += 1
            continue
        num_scored += 1
        denominator += 1
        objective_total += float(result.get("objective", 0.0))
        for name, value in _field_scores(result).items():
            field_totals[name] = field_totals.get(name, 0.0) + float(value)
            field_counts[name] = field_counts.get(name, 0) + 1

    field_accuracies = {name: field_totals[name] / field_counts[name] for name in field_totals if field_counts[name]}
    return EvalReport(
        num_cases=len(per_case),
        objective=(objective_total / denominator) if denominator else 0.0,
        field_accuracies=field_accuracies,
        field_sample_counts=dict(field_counts),
        per_case=per_case,
        num_scored=num_scored,
        num_errored=num_errored,
        num_unscoreable=num_unscoreable,
    )


def _field_scores(result: Mapping[str, Any]) -> dict[str, float]:
    scores = result.get("field_scores")
    if not isinstance(scores, Mapping):
        return {}
    return {str(name): float(value) for name, value in scores.items()}


def run_evaluation(
    *,
    agent: HotpotQAPydanticAgent,
    records: Sequence[HotpotQARecord],
    config: HotpotQAConfig | None = None,
    field_weights: Mapping[str, float] | None = None,
    max_concurrency: int = 4,
) -> EvalReport:
    """Synchronous wrapper around :func:`evaluate_agent_on_records`."""
    return asyncio.run(
        evaluate_agent_on_records(
            agent=agent,
            records=records,
            config=config,
            field_weights=field_weights,
            max_concurrency=max_concurrency,
        )
    )


__all__ = [
    "EvalReport",
    "evaluate_agent_on_records",
    "evaluate_record",
    "run_agent_on_record",
    "run_evaluation",
]
