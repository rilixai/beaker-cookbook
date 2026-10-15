"""Latency changes the hosted objective only when the task is correct."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from automationbench.schema.world import WorldState
from beaker import Case, CaseResult


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
import beaker_integration as integration  # noqa: E402


@pytest.mark.parametrize(("seconds", "objective"), [(30.0, 29 / 30), (60.0, 0.95), (120.0, 14 / 15)])
@pytest.mark.parametrize("error", [None, "agent error after required actions"])
async def test_faster_correct_tasks_improve_the_actual_score(monkeypatch, tmp_path, seconds, objective, error) -> None:
    monkeypatch.setattr(integration, "_authored_initial_states", lambda: {"sample": {"gmail": {"messages": []}}})
    case = Case(
        id="sample",
        input={"task_name": "sample"},
        expected={"assertions": [{"type": "gmail_message_sent_to", "to": "target@example.com"}]},
    )
    end = WorldState(gmail={"messages": [{"to": ["target@example.com"], "label_ids": ["SENT"]}]})
    result = CaseResult(
        output=None,
        context={
            "end_state": end.model_dump(mode="json"),
            "latency_s": seconds,
            "error": error,
        },
    )
    score = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    assert score.objective == pytest.approx(objective if error is None else 0.9)
    assert score.field_scores["partial_credit"] == score.field_scores["task_completed_correctly"] == 1.0
    assert score.field_scores["latency_score"] == pytest.approx(60 / (60 + seconds) if error is None else 0.0)
    assert all(check.verdict == "pass" for check in score.checks)


@pytest.mark.parametrize("error", [None, "rollout failed"])
async def test_incomplete_tasks_never_earn_a_speed_bonus(monkeypatch, tmp_path, error) -> None:
    monkeypatch.setattr(integration, "_authored_initial_states", lambda: {"sample": {"gmail": {"messages": []}}})
    case = Case(
        id="sample",
        input={},
        expected={
            "assertions": [
                {"type": "gmail_message_sent_to", "to": "target@example.com"},
                {"type": "gmail_message_sent_to", "to": "missing@example.com"},
            ]
        },
    )
    end = WorldState(gmail={"messages": [{"to": ["target@example.com"], "label_ids": ["SENT"]}]})
    scores = []
    for elapsed in [0.001, 120.0]:
        result = CaseResult(
            output=None,
            context={
                "end_state": end.model_dump(mode="json"),
                "latency_s": elapsed,
                "error": error,
            },
        )
        scores.append(await integration.score_case(case=case, result=result, case_files_dir=tmp_path))
    assert scores[0] == scores[1]
    assert scores[0].objective == pytest.approx(0.36)
    assert scores[0].field_scores["latency_score"] == 0.0


@pytest.mark.parametrize("elapsed", [None, False, True, 0, -1, "1", float("nan"), float("inf")])
def test_missing_or_invalid_timing_earns_no_bonus(elapsed) -> None:
    assert integration._latency_score({"latency_s": elapsed}, correct=True) == 0.0
