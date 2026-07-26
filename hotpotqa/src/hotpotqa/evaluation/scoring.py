"""Answer + retrieval scoring for HotpotQA.

Three fields are scored per case:

* ``answer`` — exact-match (0/1) after HotpotQA-standard normalization.
* ``answer_f1`` — token-level F1 (the headline HotpotQA answer metric).
* ``supporting_titles_recall`` — fraction of the gold supporting paragraph
  titles the agent retrieved at any hop. This is the multi-hop signal that
  says whether retrieval found the evidence at all.

The HotpotQA paper also reports supporting-fact F1 at the sentence level. We
intentionally score at title granularity here because the agent retrieves
paragraphs, not sentences; sentence-level supporting-fact extraction can be
layered on later without changing the objective.

The answer-string scoring functions (:func:`normalize_answer` and
:func:`f1_score`) come from :mod:`hotpotqa.data.eval`, which vendors the
canonical HotpotQA evaluator. Keep them as the single source of truth.

:func:`objective_score` collapses the per-field scores into one number using
:data:`HOTPOTQA_FIELD_WEIGHTS` — pure exact-match by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..data.dataset import HotpotQARecord
from ..data.eval import exact_match_score, f1_score, normalize_answer


ANSWER_FIELD = "answer"
ANSWER_F1_FIELD = "answer_f1"
SUPPORTING_TITLES_RECALL_FIELD = "supporting_titles_recall"


HOTPOTQA_FIELD_WEIGHTS: dict[str, float] = {
    # Pure exact match, matching the reference HotpotQA pipelines that score
    # with ``dspy.evaluate.answer_exact_match`` — the headline number is the
    # EM 0/1.
    #
    # ``answer_f1`` and ``supporting_titles_recall`` are still *computed* and
    # surfaced in the eval summary as diagnostics — they just don't move the
    # objective. Pass different ``field_weights`` for a blended ablation.
    ANSWER_FIELD: 1.0,
    ANSWER_F1_FIELD: 0.0,
    SUPPORTING_TITLES_RECALL_FIELD: 0.0,
}


def supporting_titles_recall(retrieved_titles: Sequence[str], gold_titles: Sequence[str]) -> float:
    """Fraction of ``gold_titles`` present in ``retrieved_titles`` (case-insensitive)."""
    if not gold_titles:
        return 1.0
    retrieved = {t.strip().lower() for t in retrieved_titles if t.strip()}
    if not retrieved:
        return 0.0
    hits = sum(1 for gold in gold_titles if gold.strip().lower() in retrieved)
    return hits / len(gold_titles)


def score_prediction(
    *,
    record: HotpotQARecord,
    answer: str,
    retrieved_titles: Sequence[str],
) -> dict[str, float]:
    """Score one agent prediction against ``record``'s gold answer + titles.

    A field is only scored when the record carries supervision for it: a case
    with an empty gold answer contributes no ``answer`` / ``answer_f1`` score,
    and one with no gold supporting titles contributes no recall score. An
    entirely unsupervised record therefore scores ``{}`` — the caller treats
    that as *unscoreable* rather than as a zero.
    """
    field_scores: dict[str, float] = {}
    if record.answer.strip():
        field_scores[ANSWER_FIELD] = 1.0 if exact_match_score(answer, record.answer) else 0.0
        field_scores[ANSWER_F1_FIELD] = _clamp_unit(f1_score(answer, record.answer))
    if record.supporting_titles:
        field_scores[SUPPORTING_TITLES_RECALL_FIELD] = _clamp_unit(
            supporting_titles_recall(retrieved_titles, record.supporting_titles)
        )
    return field_scores


def objective_score(
    field_scores: Mapping[str, float],
    field_weights: Mapping[str, float] | None = None,
) -> float:
    """Collapse per-field scores into one weighted objective in ``[0, 1]``.

    Weights are normalized over the (positive) entries of ``field_weights``;
    a weighted field the case did not score counts as ``0``. With no weights
    every scored field is weighted equally.
    """
    weights = _normalized_weights(field_scores=field_scores, field_weights=field_weights)
    if not weights:
        return 0.0
    return sum(weight * _clamp_unit(field_scores.get(name, 0.0)) for name, weight in weights.items())


def _normalized_weights(
    *,
    field_scores: Mapping[str, float],
    field_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if field_weights:
        raw = {name: float(weight) for name, weight in field_weights.items() if float(weight) > 0.0}
    else:
        raw = dict.fromkeys(field_scores, 1.0)
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {name: weight / total for name, weight in raw.items()}


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "f1_score",
    "normalize_answer",
    "objective_score",
    "score_prediction",
    "supporting_titles_recall",
]
