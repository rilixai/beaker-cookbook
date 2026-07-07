"""Field scoring and the rilixai :class:`~rilixai.CaseScorer` for HotpotQA.

Three optimization fields are exposed:

* ``answer`` — exact-match (0/1) after HotpotQA-standard normalization.
* ``answer_f1`` — token-level F1 (the headline HotpotQA answer metric).
* ``supporting_titles_recall`` — fraction of gold supporting paragraph titles
  the pipeline retrieved at any hop. This is the multi-hop signal the
  reflection LM uses to learn when retrieval is missing hops.

The HotpotQA paper also reports supporting-fact F1 at the sentence level. We
intentionally score at title granularity here because the pipeline retrieves
paragraphs, not sentences; sentence-level supporting-fact extraction can be
layered on later without changing the optimization objective.

The answer-string scoring functions (``normalize_answer`` and ``f1_score``)
are re-exported from :mod:`hotpotqa.data.eval`, which vendors the
canonical HotpotQA evaluator. Keep them as the single source of truth.

The ``run_case`` runs the agent and stashes the model answer + retrieved
titles on ``result.output``; this scorer reads them back, scores the three
fields against ``case.ground_truth``, and collapses the weighted objective —
mirroring the SDK ``CaseScorer`` protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rilixai import Case, CaseResult, CaseScore, objective_score

from ..data.eval import exact_match_score, f1_score, normalize_answer


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "HotpotQAScorer",
    "f1_score",
    "normalize_answer",
]


ANSWER_FIELD = "answer"
ANSWER_F1_FIELD = "answer_f1"
SUPPORTING_TITLES_RECALL_FIELD = "supporting_titles_recall"

# The field the run_case stashes the retrieved paragraph titles under.
RETRIEVED_TITLES_KEY = "retrieved_titles"


HOTPOTQA_FIELD_WEIGHTS: dict[str, float] = {
    # Paper-faithful pure exact-match. The GEPA artifact registers
    # HotpotQA with ``dspy.evaluate.answer_exact_match`` at ``frac=1.0``
    # — every per-case reward GEPA sees is the EM 0/1. We match that here
    # so optimize-time selection pressure is identical to the paper's.
    #
    # ``answer_f1`` and ``supporting_titles_recall`` are still *computed*
    # and surfaced in the eval summary as diagnostics — they just don't
    # influence candidate selection. To run the diagnostics-blended
    # variant for ablations, override ``field_weights`` when calling
    # :func:`build_hotpotqa_spec`.
    ANSWER_FIELD: 1.0,
    ANSWER_F1_FIELD: 0.0,
    SUPPORTING_TITLES_RECALL_FIELD: 0.0,
}


def _exact_match(predicted: Any, actual: Any) -> float:
    return 1.0 if exact_match_score(predicted, actual) else 0.0


def _supporting_titles_recall(predicted: Any, actual: Any) -> float:
    gold_titles = _coerce_title_sequence(actual)
    if not gold_titles:
        return 1.0
    retrieved_titles = {t.strip().lower() for t in _coerce_title_sequence(predicted) if t and t.strip()}
    if not retrieved_titles:
        return 0.0
    hits = sum(1 for gold in gold_titles if gold.strip().lower() in retrieved_titles)
    return hits / len(gold_titles)


def _coerce_title_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        # Permit the HF ``supporting_facts`` shape ``{"title": [...], "sent_id": [...]}``.
        titles = value.get("title")
        if isinstance(titles, Sequence) and not isinstance(titles, (str, bytes, bytearray)):
            return [str(t) for t in titles]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(t) for t in value]
    return [str(value)]


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


class HotpotQAScorer:
    """rilixai :class:`~rilixai.CaseScorer` for the three HotpotQA fields.

    Scores the model answer (EM + token F1) and the retrieval trace
    (supporting-title recall) off ``result.output`` against the gold
    ``answer`` + ``supporting_titles`` bundled on ``case.ground_truth``.
    The weighted objective collapses to pure exact-match by default (see
    :data:`HOTPOTQA_FIELD_WEIGHTS`); override ``field_weights`` for the
    diagnostics-blended ablation.
    """

    def __init__(self, field_weights: Mapping[str, float] | None = None) -> None:
        self.field_weights: dict[str, float] = dict(field_weights or HOTPOTQA_FIELD_WEIGHTS)

    async def score_case(self, *, case: Case, result: CaseResult) -> CaseScore:
        output = result.output if isinstance(result.output, Mapping) else {}
        ground_truth = case.ground_truth if isinstance(case.ground_truth, Mapping) else {}

        predicted_answer = output.get(ANSWER_FIELD)
        gold_answer = ground_truth.get(ANSWER_FIELD)
        retrieved_titles = output.get(RETRIEVED_TITLES_KEY)
        gold_titles = ground_truth.get("supporting_titles")

        field_scores: dict[str, float] = {}
        if gold_answer is not None and str(gold_answer).strip() != "":
            field_scores[ANSWER_FIELD] = _clamp_unit(_safe(_exact_match, predicted_answer, gold_answer))
            field_scores[ANSWER_F1_FIELD] = _clamp_unit(_safe(f1_score, predicted_answer, gold_answer))
        if _coerce_title_sequence(gold_titles):
            field_scores[SUPPORTING_TITLES_RECALL_FIELD] = _clamp_unit(
                _safe(_supporting_titles_recall, retrieved_titles, gold_titles)
            )

        return CaseScore(
            field_scores=field_scores,
            objective=objective_score(field_scores, field_weights=self.field_weights),
            key=ANSWER_FIELD,
        )


def _safe(fn: Any, predicted: Any, actual: Any) -> float:
    try:
        return float(fn(predicted, actual))
    except Exception:
        return 0.0
