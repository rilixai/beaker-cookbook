"""Score one question with upstream MMDM rules, and summarize a run. Fixed.

Per-question scores come from ``mmdm_eval.metrics.score`` (which uses
``set_metrics.evaluate_hard_set`` for set questions). Nothing here reimplements
upstream scoring; this module maps its record onto Beaker's objective and fields.

Objective by label type:
  hard single: 1 if the unique top option is correct (a tie for first is wrong)
  hard set:    1 if every gold option ranks strictly above every other option
  soft:        1 - total variation
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mmdm_eval.common import valid_p
from mmdm_eval.metrics import label_kind
from mmdm_eval.metrics import score as upstream_score
from mmdm_eval.metrics import summarize as upstream_summarize
from mmdm_eval.scoring import score_predictions

from mmdm.data import Expected, Question


TIE_TOLERANCE = 1e-12  # upstream's tolerance for a tie at the top
ECE_BINS = 10
BRIER_MAX = 2.0  # upstream Brier is a sum of squared errors over options


def upstream_row(case_id: str, question: Question, expected: Expected) -> dict[str, Any]:
    """The question and its labels in upstream's raw-row shape."""
    return {
        "id": case_id,
        **question.model_dump(mode="json"),
        "gold": list(expected.probabilities),
        "correct_option_ids": list(expected.correct_option_ids),
        "label_semantics": expected.label_semantics,
        "task_category": expected.task_category,
        "task_subcategory": expected.task_subcategory,
        "source": expected.source,
    }


@dataclass(frozen=True)
class Scored:
    objective: float
    field_scores: dict[str, float]
    kind: str  # "hard", "hard_set" or "soft", as upstream labels them
    error: str | None
    expected_top: list[str]
    predicted_top: list[str] | None
    upstream: dict[str, Any] = field(repr=False)

    @property
    def valid(self) -> bool:
        return self.error is None


def _top(ids: Sequence[str], p: Sequence[float], k: int | None) -> list[str]:
    """Top-k by probability (option order breaks ties, as upstream ranks); k=None: every option tied for first."""
    if k is None:
        best = max(p)
        return [i for i, x in zip(ids, p, strict=True) if abs(x - best) <= TIE_TOLERANCE]
    order = sorted(range(len(ids)), key=lambda i: (-p[i], i))
    return [ids[i] for i in order[:k]]


def score(case_id: str, question: Question, expected: Expected, output: Any) -> Scored:
    row = upstream_row(case_id, question, expected)
    kind = label_kind(row)
    ids = question.option_ids
    names = {o.id: o.name for o in question.options}
    record: dict[str, Any] = {"ok": False, "mode": "native", "status": "request_error"}
    output = output if isinstance(output, Mapping) else {}
    error = output.get("error") if output else "no output"
    p: list[float] | None = None
    if error is None:
        probabilities = output.get("probabilities")
        try:
            if not isinstance(probabilities, Mapping) or set(probabilities) != set(ids):
                raise ValueError("missing_option")
            p = valid_p([probabilities[i] for i in ids], len(ids))
            record = {"ok": True, "mode": "native", "status": "complete", "probabilities": p}
        except (ValueError, TypeError, AssertionError, KeyError) as exc:
            error = "missing_option" if str(exc) == "missing_option" else "malformed_answer"
            record["status"] = "invalid_prediction"
    upstream = upstream_score(row, record)

    gold = list(expected.probabilities)
    k = len(expected.correct_option_ids) if kind == "hard_set" else None
    expected_ids = list(expected.correct_option_ids) if kind == "hard_set" else _top(ids, gold, None)
    predicted_ids = _top(ids, p, k) if p is not None else None

    fields: dict[str, float] = {}
    if kind == "hard":
        fields["hard_accuracy"] = float(upstream["correct"] or 0.0)
    elif kind == "hard_set":
        fields["set_match"] = float(upstream["correct"] or 0.0)
    if p is not None:
        # Upstream's multi-class Brier is a sum of squared errors in [0, 2]; field scores must be in [0, 1].
        fields["brier"] = float(upstream["brier"]) / BRIER_MAX
        fields["total_variation"] = float(upstream["tv"])
        fields["top_probability"] = max(p)
        if upstream["score_normalized_mae"] is not None:
            fields["score_nmae"] = float(upstream["score_normalized_mae"])
        objective = 1.0 - fields["total_variation"] if kind == "soft" else float(upstream["correct"])
    else:
        objective = 0.0
    return Scored(
        objective=objective,
        field_scores=fields,
        kind=kind,
        error=None if error is None else str(error),
        expected_top=[names[i] for i in expected_ids],
        predicted_top=None if predicted_ids is None else [names[i] for i in predicted_ids],
        upstream=upstream,
    )


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return math.fsum(items) / len(items) if items else None


def _upstream_brier(cases: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean Brier on upstream's [0, 2] scale, so it matches ``mmdm-score``."""
    mean = _mean(c["fields"]["brier"] for c in cases)
    return None if mean is None else mean * BRIER_MAX


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def expected_calibration_error(pairs: Sequence[tuple[float, float]], bins: int = ECE_BINS) -> float | None:
    """Equal-width ECE over (top probability, correct) pairs."""
    if not pairs:
        return None
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for confidence, correct in pairs:
        grouped[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    return math.fsum(
        len(members) / len(pairs) * abs(_mean(c for _, c in members) - _mean(p for p, _ in members))  # type: ignore[operator]
        for members in grouped.values()
    )


def summarize(cases: Sequence[Mapping[str, Any]], *, price_per_million_input_tokens: float | None) -> dict[str, Any]:
    """Run-level metrics from per-case records (see cli.evaluate for the record shape)."""
    valid = [c for c in cases if c["error"] is None]
    hard = [c for c in cases if c["kind"] == "hard"]
    sets = [c for c in cases if c["kind"] == "hard_set"]
    soft_valid = [c for c in valid if c["kind"] == "soft"]
    score_valid = [c for c in valid if c["task"] == "score"]
    calibration = [
        (c["fields"]["top_probability"], c["fields"]["hard_accuracy"]) for c in valid if c["kind"] == "hard"
    ]
    categories: dict[str, Any] = {}
    for category in sorted({c["category"] for c in cases}):
        members = [c for c in cases if c["category"] == category]
        cat_hard = [c for c in members if c["kind"] == "hard"]
        cat_sets = [c for c in members if c["kind"] == "hard_set"]
        cat_soft = [c for c in members if c["kind"] == "soft" and c["error"] is None]
        categories[category] = {
            "cases": len(members),
            "failed": sum(c["error"] is not None for c in members),
            "hard_single": len(cat_hard),
            "hard_accuracy": _mean(c["fields"].get("hard_accuracy", 0.0) for c in cat_hard),
            "hard_set": len(cat_sets),
            "set_match": _mean(c["fields"].get("set_match", 0.0) for c in cat_sets),
            "soft": len(cat_soft),
            "soft_brier": _upstream_brier(cat_soft),
            "objective": _mean(c["objective"] for c in members),
        }
    latencies = [c["latency_s"] for c in cases]
    tokens = [c["decision_input_tokens"] for c in cases]
    mean_tokens = _mean(tokens)
    return {
        "cases": len(cases),
        "objective": _mean(c["objective"] for c in cases),
        "hard_accuracy": _mean(c["fields"].get("hard_accuracy", 0.0) for c in hard),
        "hard_single_cases": len(hard),
        "set_match": _mean(c["fields"].get("set_match", 0.0) for c in sets),
        "hard_set_cases": len(sets),
        "soft_brier": _upstream_brier(soft_valid),
        "soft_total_variation": _mean(c["fields"]["total_variation"] for c in soft_valid),
        "soft_cases_scored": len(soft_valid),
        "score_nmae": _mean(c["fields"]["score_nmae"] for c in score_valid),
        "score_cases_scored": len(score_valid),
        "ece_hard_single": expected_calibration_error(calibration),
        "coverage": {
            "native": len(valid),
            "grouped": 0,
            "failed": len(cases) - len(valid),
            "failed_by_status": dict(Counter(str(c["error"]).split(":")[0] for c in cases if c["error"] is not None)),
        },
        "latency_s": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "decision_input_tokens_per_question": mean_tokens,
        "cost_per_1000_questions_usd": (
            None
            if mean_tokens is None or price_per_million_input_tokens is None
            else mean_tokens * 1000 * price_per_million_input_tokens / 1e6
        ),
        "categories": categories,
    }


def upstream_prediction(case_id: str, question: Question, output: Mapping[str, Any]) -> dict[str, Any]:
    """One line of an ``mmdm-score --predictions`` file (upstream integer IDs, option order)."""
    upstream_id = int(case_id.removeprefix("mmdm-"))
    probabilities = output.get("probabilities")
    if output.get("error") is not None or not isinstance(probabilities, Mapping):
        return {"id": upstream_id, "ok": False, "status": "request_error"}
    if set(probabilities) != set(question.option_ids):
        return {"id": upstream_id, "probabilities": []}  # upstream records this as invalid_prediction
    return {"id": upstream_id, "probabilities": [probabilities[i] for i in question.option_ids]}


def upstream_cross_check(
    rows: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], ours: Mapping[str, Any]
) -> dict[str, Any]:
    """Score the predictions file with upstream ``score_predictions`` and compare with our summary."""
    data = [{**row, "id": int(str(row["id"]).removeprefix("mmdm-"))} for row in rows]
    theirs = upstream_summarize(score_predictions(data, predictions))
    pairs = {
        "hard_accuracy": (theirs["hard_single"]["accuracy_all"], ours["hard_accuracy"]),
        "set_match": (theirs["hard_set"]["accuracy_all"], ours["set_match"]),
        "soft_brier": (theirs["soft_brier"]["mean"], ours["soft_brier"]),
        "score_nmae": (theirs["score_normalized_mae"]["mean"], ours["score_nmae"]),
        "valid": (theirs["valid"], ours["coverage"]["native"]),
    }
    mismatches = {
        name: {"upstream": a, "recipe": b}
        for name, (a, b) in pairs.items()
        if not (a is None and b is None)
        and (a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    }
    return {"matches": not mismatches, "mismatches": mismatches, "upstream_overall": theirs}
