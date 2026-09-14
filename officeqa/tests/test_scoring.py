"""Upstream scorer contract (copied verbatim) and our zero-on-failure wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from officeqa import config
from officeqa.evaluation.reward import score_answer
from officeqa.runner import score_all


REWARD = Path(__file__).resolve().parents[1] / "src" / "officeqa" / "evaluation" / "reward.py"


def test_reward_provenance_header() -> None:
    head = REWARD.read_text(encoding="utf-8").splitlines()[:5]
    assert head[0].startswith("# Copied verbatim from https://github.com/databricks/officeqa")
    assert any("Commit:" in h for h in head) and any("Apache" in h for h in head)


@pytest.mark.parametrize("tol", config.TOLERANCES)
def test_spec_examples_at_every_tolerance(tol: float) -> None:
    assert score_answer(ground_truth="123.45", predicted="123.45", tolerance=tol) == 1.0
    assert score_answer(ground_truth="$7,046,001.98", predicted="$7,046,001.98", tolerance=tol) == 1.0
    assert (
        score_answer(
            ground_truth="[Massachusetts, 0.866]", predicted="Massachusetts, with a ratio of 0.866", tolerance=tol
        )
        == 1.0
    )


def test_tolerances_are_the_headline_grid() -> None:
    assert config.TOLERANCES == (0.0, 0.001, 0.01, 0.05)
    assert config.HEADLINE_TOLERANCE == 0.0


def test_tolerance_semantics_relative_error() -> None:
    gt = "1000"
    assert score_answer(gt, "1000", 0.0) == 1.0
    assert score_answer(gt, "1000.5", 0.0) == 0.0
    assert score_answer(gt, "1000.5", 0.001) == 1.0  # 0.05% error
    assert score_answer(gt, "1005", 0.001) == 0.0
    assert score_answer(gt, "1005", 0.01) == 1.0
    assert score_answer(gt, "1040", 0.01) == 0.0
    assert score_answer(gt, "1040", 0.05) == 1.0
    assert score_answer(gt, "1100", 0.05) == 0.0


def test_numeric_formats() -> None:
    assert score_answer("7046001.98", "$7,046,001.98", 0.0) == 1.0
    assert score_answer("$7,046,001.98", "7046001.98", 0.0) == 1.0
    assert score_answer("12.5%", "12.5", 0.0) == 1.0
    assert score_answer("123.45", "<FINAL_ANSWER>123.45</FINAL_ANSWER>", 0.0) == 1.0
    assert score_answer("123.45", "543.21", 0.05) == 0.0


def test_bracketed_lists() -> None:
    assert score_answer("[Massachusetts, 0.866]", "[Massachusetts, 0.866]", 0.0) == 1.0
    assert score_answer("[Massachusetts, 0.866]", "Texas, 0.866", 0.0) == 0.0
    assert score_answer("[Massachusetts, 0.866]", "Massachusetts, 0.9", 0.0) == 0.0


def test_missing_final_answer_scores_zero_everywhere() -> None:
    assert score_all("123.45", None) == {"0.0": 0.0, "0.001": 0.0, "0.01": 0.0, "0.05": 0.0}
    assert score_all("123.45", "   ") == {"0.0": 0.0, "0.001": 0.0, "0.01": 0.0, "0.05": 0.0}
    assert score_all("123.45", "123.5") == {"0.0": 0.0, "0.001": 1.0, "0.01": 1.0, "0.05": 1.0}
