"""Scoring follows upstream rules on hand-made cases."""

from __future__ import annotations

from typing import Any

import pytest
from mmdm_eval.set_metrics import evaluate_hard_set

from mmdm.data import Expected, Option, Question
from mmdm.scoring import expected_calibration_error, score


def case(
    gold: list[float],
    semantics: str,
    *,
    correct: list[str] | None = None,
    task: str = "choice",
    score_values: tuple[int, ...] | None = None,
) -> tuple[Question, Expected]:
    ids = [f"o{i}" for i in range(len(gold))]
    question = Question(
        state="s",
        question="q",
        options=tuple(Option(id=i, name=f"Name {i}") for i in ids),
        task=task,  # type: ignore[arg-type]
        score_values=score_values,
    )
    expected = Expected(
        label_semantics=semantics,  # type: ignore[arg-type]
        probabilities=tuple(gold),
        correct_option_ids=tuple(correct or [ids[gold.index(max(gold))]]),
        task_category="intent_routing",
        task_subcategory="synthetic",
        source="mmdm",
    )
    return question, expected


def output(*p: float) -> dict[str, Any]:
    return {"probabilities": {f"o{i}": x for i, x in enumerate(p)}, "error": None}


def test_hard_single_needs_a_unique_correct_top_option() -> None:
    q, e = case([1.0, 0.0, 0.0], "categorical_hard")
    tied = score("mmdm-1", q, e, output(0.4, 0.4, 0.2))
    assert tied.objective == 0.0 and tied.field_scores["hard_accuracy"] == 0.0
    assert tied.predicted_top == ["Name o0", "Name o1"]
    right = score("mmdm-1", q, e, output(0.5, 0.3, 0.2))
    assert right.objective == 1.0 and right.expected_top == ["Name o0"] and right.predicted_top == ["Name o0"]
    assert right.field_scores["top_probability"] == 0.5
    assert right.field_scores["brier"] == pytest.approx(0.25 + 0.09 + 0.04)


def test_set_match_fails_on_a_tie_at_the_boundary() -> None:
    q, e = case([0.5, 0.5, 0.0, 0.0], "set_hard", correct=["o0", "o1"], task="multichoice")
    boundary_tie = (0.3, 0.25, 0.25, 0.2)
    strict = (0.3, 0.3, 0.2, 0.2)
    for p, want in ((boundary_tie, 0.0), (strict, 1.0)):
        scored = score("mmdm-2", q, e, output(*p))
        upstream = evaluate_hard_set(["o0", "o1", "o2", "o3"], ["o0", "o1"], list(p))["topn_all_hit"]
        assert scored.objective == scored.field_scores["set_match"] == upstream == want
    assert score("mmdm-2", q, e, output(*strict)).predicted_top == ["Name o0", "Name o1"]


def test_soft_distribution_scores_one_minus_total_variation() -> None:
    q, e = case([0.5, 0.3, 0.2], "categorical_soft")
    scored = score("mmdm-3", q, e, output(0.6, 0.2, 0.2))
    assert scored.kind == "soft"
    assert scored.field_scores["total_variation"] == pytest.approx(0.1)
    assert scored.objective == pytest.approx(0.9)
    assert scored.field_scores["brier"] == pytest.approx(0.02)
    assert "hard_accuracy" not in scored.field_scores and "score_nmae" not in scored.field_scores


def test_score_task_reports_normalized_mae() -> None:
    q, e = case([0.0, 0.5, 0.5], "stochastic_ordinal_score", task="score", score_values=(1, 2, 3))
    scored = score("mmdm-4", q, e, output(0.2, 0.4, 0.4))
    # |sum((p - q) * value)| / (max - min) = |0.2 - 0.2 - 0.3| / 2
    assert scored.field_scores["score_nmae"] == pytest.approx(0.15)
    assert scored.objective == pytest.approx(1 - 0.2)


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({"probabilities": None, "error": "request_error: TypeSafeBadRequestError"}, "request_error"),
        ({"probabilities": {"o0": 1.0, "o1": 0.0}, "error": None}, "missing_option"),
        ({"probabilities": {"o0": 0.9, "o1": 0.9, "o2": 0.9}, "error": None}, "malformed_answer"),
        (None, "no output"),
    ],
)
def test_failures_score_zero_with_an_observable_error(result: Any, error: str) -> None:
    q, e = case([1.0, 0.0, 0.0], "categorical_hard")
    scored = score("mmdm-5", q, e, result)
    assert scored.objective == 0.0 and scored.field_scores == {"hard_accuracy": 0.0}
    assert scored.error is not None and scored.error.startswith(error)
    assert scored.predicted_top is None


def test_expected_calibration_error() -> None:
    assert expected_calibration_error([(0.9, 1.0), (0.9, 0.0)]) == pytest.approx(0.4)
    assert expected_calibration_error([(0.25, 0.0), (0.95, 1.0)]) == pytest.approx(0.5 * 0.25 + 0.5 * 0.05)
    assert expected_calibration_error([]) is None
