"""Regression coverage for condition Checks and shared observed OUTPUT."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from automationbench.rubric import partial_credit
from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState
from beaker import Case, CaseResult


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
import beaker_integration as integration  # noqa: E402
from assertion_diagnostics import capture_assertion_returns, failure_trace  # noqa: E402
from world_diff import OUTPUT_MAX_BYTES, ServiceDiffs  # noqa: E402


@pytest.mark.parametrize(("passed", "excluded"), [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize(
    ("assertion_type", "params"),
    [
        ("salesforce_campaign_member_exists", {"contact_id": "c1", "campaign_id": "p1"}),
        ("gmail_draft_body_contains", {"to": "c1", "body_contains": "p1"}),
        ("gmail_message_sent_to_with_body_contains", {"to": "c1", "body_contains": ["p1"]}),
        ("slack_message_exists", {"channel": "c1", "text_contains": "p1"}),
        ("google_sheets_cell_equals", {"spreadsheet_id": "c1", "value": "p1"}),
    ],
)
def test_all_conditions_preserve_verdict_exclusion_and_parameters(
    passed: bool, excluded: bool, assertion_type, params
) -> None:
    check = integration._check(
        {
            "type": assertion_type,
            "passed": passed,
            "excluded": excluded,
            "params": {**params, "scored": not excluded},
        },
        error=None,
        entities={},
        world=WorldState(),
    )
    assert check.verdict == ("pass" if passed else "fail") and check.informational == excluded
    assert check.expected is None and check.predicted is None
    assert '"c1"' in check.description and "scored" not in check.description
    if excluded:
        assert check.message == "excluded from scoring by the task author"
    elif passed:
        assert check.message is None
    else:
        assert check.message == "not satisfied by the end state; see OUTPUT for observed app changes"


async def test_run_case_displays_changes_once_and_native_scoring_uses_full_context(monkeypatch, tmp_path) -> None:
    initial = {"gmail": {"messages": []}}
    end = WorldState(
        gmail={"messages": [{"to": ["target@example.com"], "label_ids": ["SENT"], "body_plain": "Total: $19,500"}]}
    ).model_dump(mode="json")
    monkeypatch.setattr(integration, "_authored_initial_states", lambda: {"sample": initial})
    monkeypatch.setattr(integration, "_sample_for", lambda _: SimpleNamespace(task_name="sample", domain="finance"))
    monkeypatch.setattr(integration, "_candidate_root", lambda: tmp_path)
    client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(integration, "_client_for", lambda _: (client, None))
    monkeypatch.setattr(
        integration, "run_rollout_raw", AsyncMock(return_value={"completion": ["done"], "_end_state": end})
    )

    @contextmanager
    def stage(*args, **kwargs):
        yield SimpleNamespace(output=lambda _: None)

    result = await integration.run_case(case_input={}, runtime=SimpleNamespace(trace=SimpleNamespace(stage=stage)))
    assert result.output_kind == "record" and result.context["end_state"] == end
    assert result.output["gmail.added"][0]["record"]["body_plain"] == "Total: $19,500"
    client.close.assert_awaited_once()
    case = Case(
        id="sample",
        input={"task_name": "sample"},
        expected={
            "assertions": [
                {"type": "gmail_message_sent_to", "to": "target@example.com"},
                {"type": "gmail_message_sent_to", "to": "missing@example.com"},
                {
                    "type": "gmail_message_sent_to_with_body_contains",
                    "to": "target@example.com",
                    "body_contains": ["19500"],
                },
            ]
        },
    )
    score = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    assert score.field_scores == {"partial_credit": pytest.approx(2 / 3), "task_completed_correctly": 0.0}
    assert score.objective == pytest.approx(0.8 * 2 / 3)
    assert [check.verdict for check in score.checks] == ["pass", "fail", "pass"]
    assert all(check.expected is None and check.predicted is None for check in score.checks)
    assert "Condition gmail_message_sent_to evaluated to False" in score.checks[1].message
    assert '"expected_to": "missing@example.com"' in score.checks[1].message
    assert '"all_recipients"' in score.checks[1].message
    assert score == await integration.score_case(
        case=case, result=CaseResult(output=None, context=result.context), case_files_dir=tmp_path
    )
    monkeypatch.setattr(integration, "run_rollout_raw", AsyncMock(return_value={"completion": ["done"]}))
    missing = await integration.run_case(case_input={}, runtime=SimpleNamespace(trace=SimpleNamespace(stage=stage)))
    assert missing.output is None and missing.output_kind == "none"


def test_output_keeps_all_change_kinds_and_marks_size_limits(monkeypatch) -> None:
    raw = {"gmail": {"messages": [{"id": "change", "subject": "Before"}, {"id": "remove"}]}}
    end = WorldState(
        gmail={"messages": [{"id": "change", "subject": "After"}, {"id": "add", "body_plain": "new"}]},
        slack={"messages": [{"id": "slack", "channel_id": "c", "user_id": "u", "text": "Unasserted change"}]},
    ).model_dump(mode="json")
    output = ServiceDiffs(raw, WorldState(**raw), end).to_output()
    assert output["gmail.added"][0]["record"]["id"] == "add"
    assert output["gmail.changed"][0]["after"]["subject"] == "After"
    assert output["gmail.removed"][0]["record"]["id"] == "remove"
    assert output["slack.added"][0]["record"]["text"] == "Unasserted change"
    assert ServiceDiffs(raw, WorldState(**raw), WorldState(**raw).model_dump(mode="json")).to_output() == {
        "summary": "No app changes observed."
    }
    diffs = ServiceDiffs({}, None, {})
    services = ["gmail", "slack", "salesforce", "zendesk", "google_sheets", "google_drive", "jira"]
    monkeypatch.setattr(
        diffs,
        "get",
        lambda service: (
            {"service": service, "added": [{"record": {"text": "界" * 1000}}]} if service in services else None
        ),
    )
    output = diffs.to_output()
    assert len(json.dumps(output).encode("utf-8")) <= OUTPUT_MAX_BYTES and output["truncated"] is True
    assert {key.split(".")[0] for key in output if key != "truncated"} == set(services)


def test_native_failure_diagnostics_across_apps_without_rerunning_assertions(monkeypatch) -> None:
    world = WorldState(
        gmail={"messages": [{"to": ["a@example.com"], "label_ids": ["SENT"], "body_plain": "Total: $19,200"}]},
        slack={"messages": [{"id": "m", "user_id": "u", "channel_id": "c", "text": "Total: $19,100"}]},
        google_sheets={
            "rows": [
                {
                    "spreadsheet_id": "s",
                    "worksheet_id": "w",
                    "row_id": 1,
                    "cells": {"department": "Engineering", "total": "19200"},
                }
            ]
        },
    )
    assertions = [
        {"type": "gmail_message_sent_to_with_body_contains", "to": "a@example.com", "body_contains": ["19,500"]},
        {"type": "slack_message_exists", "channel": "c", "text_contains": "19,500"},
        {
            "type": "google_sheets_cell_equals",
            "spreadsheet_id": "s",
            "worksheet_id": "w",
            "lookup_column": "department",
            "lookup_value": "Engineering",
            "target_column": "total",
            "expected_value": "19500",
        },
    ]
    calls = []
    original = AssertionRegistry.check

    def counted(world, assertion):
        calls.append(id(assertion))
        return original(world, assertion)

    monkeypatch.setattr(AssertionRegistry, "check", counted)
    before = world.model_dump(mode="json")
    state = {"info": {"assertions": assertions}, "world": world}
    with capture_assertion_returns(world, assertions) as events:
        assert partial_credit(state) == 0.0
    assert calls == [id(assertion) for assertion in assertions]
    assert before == world.model_dump(mode="json") and sys.getprofile() is None
    messages = [failure_trace(assertion, events[id(assertion)]) for assertion in assertions]
    assert "19200" in messages[0] and "19500" in messages[0]
    assert "19100" in messages[1] and "19500" in messages[1]
    assert '"actual": "19200"' in messages[2] and '"expected": "19500"' in messages[2]
    assert "19100" not in messages[0] and "a@example.com" not in messages[1]
    assert all(len(message.encode("utf-8")) <= 6_000 for message in messages)


def test_diagnostics_ignore_initial_state_and_restore_profiler_on_error() -> None:
    initial = WorldState(gmail={"messages": [{"to": ["initial@example.com"], "label_ids": ["SENT"]}]})
    world = WorldState()
    assertion = {"type": "gmail_message_sent_to", "to": "missing@example.com"}
    with capture_assertion_returns(world, [assertion]) as events:
        AssertionRegistry.check(initial, assertion)
        AssertionRegistry.check(world, assertion)
    message = failure_trace(assertion, events[id(assertion)])
    assert "initial@example.com" not in message
    with pytest.raises(RuntimeError), capture_assertion_returns(world, [assertion]):
        raise RuntimeError("rubric failure")
    assert sys.getprofile() is None

    def previous(*args):
        pass

    sys.setprofile(previous)
    try:
        with capture_assertion_returns(world, [assertion]) as events:
            AssertionRegistry.check(world, assertion)
        assert sys.getprofile() is previous and not events
    finally:
        sys.setprofile(None)


def test_negative_alias_and_large_unicode_diagnostics_are_bounded() -> None:
    assertion = {
        "type": "gmail_message_not_sent_to_with_body_contains",
        "to": "a@example.com",
        "body_contains": ["forbidden"],
    }
    world = WorldState(
        gmail={
            "messages": [{"to": ["a@example.com"], "label_ids": ["SENT"], "body_plain": "forbidden " + "界" * 10000}]
        }
    )
    with capture_assertion_returns(world, [assertion]) as events:
        assert not AssertionRegistry.check(world, assertion)
    message = failure_trace(assertion, events[id(assertion)])
    assert "forbidden" in message and "returned False" in message and "truncated" in message
    assert len(message.encode("utf-8")) <= 6_000
