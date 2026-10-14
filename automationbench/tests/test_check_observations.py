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


@pytest.mark.parametrize(
    "value",
    [
        "界" * 80 + " final requirement",
        ["界" * 80, "final requirement"],
        {"text": "界" * 80, "suffix": "final requirement"},
    ],
)
def test_check_names_preserve_complete_parameter_values(value) -> None:
    check = integration._check(
        {
            "type": "slack_message_exists",
            "passed": False,
            "params": {"text_contains": value, "scored": True, "excluded": False},
        },
        error=None,
        entities={},
    )
    assert "界" * 80 in check.name and "final requirement" in check.name
    assert "…" not in check.name and "scored=" not in check.name and "excluded=" not in check.name
    assert json.loads(check.description.split("Condition parameters: ", 1)[1]) == {"text_contains": value}


def test_long_entity_labels_and_distinct_requirement_suffixes_survive_in_names() -> None:
    label = "Channel " + "a" * 80 + " final label"
    prefix = "Shared requirement " + "b" * 80
    checks = [
        integration._check(
            {
                "type": "slack_message_exists",
                "passed": True,
                "params": {"channel": "c1", "text_contains": prefix + suffix},
            },
            error=None,
            entities={"c1": label},
        )
        for suffix in (" first", " second")
    ]
    assert checks[0].name != checks[1].name
    assert all(f"channel={label}" in check.name for check in checks)
    assert checks[0].name.endswith(" first") and checks[1].name.endswith(" second")
    assert all('"channel": "c1"' in check.description for check in checks)
    assert all(check.description.startswith("Check") for check in checks)


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
    assert "Required conditions:" in score.checks[1].message
    assert 'Expected to: "missing@example.com"' in score.checks[1].message
    assert "All recipients:" in score.checks[1].message
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
    assert '"19200" compared with "19500": no match' in messages[2]
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
    assert 'compared with "forbidden": match' in message and "truncated" in message
    assert "overall failure" in message and "```python" not in message
    assert len(message.encode("utf-8")) <= 6_000


@pytest.mark.parametrize("row_count", [4, 12])
def test_row_failure_summary_shows_partial_comparisons_and_accurate_row_count(row_count) -> None:
    assertion = {
        "type": "google_sheets_row_exists",
        "spreadsheet_id": "ss_inv_tracker",
        "worksheet_id": "ws_pending",
        "cell_contains": {"Invoice": "GL-88210", "Amount": "11,340.5", "Notes": "REVIEW"},
    }
    invoices = ["GL-88210"] + ["OTHER"] * (row_count - 3) + ["SP-0091", "AS-7744"]
    world = WorldState(
        google_sheets={
            "rows": [
                {
                    "spreadsheet_id": "ss_inv_tracker",
                    "worksheet_id": "ws_pending",
                    "row_id": index,
                    "cells": {"Invoice": invoice, "Amount": "wrong amount", "Notes": "REVIEW"},
                }
                for index, invoice in enumerate(invoices)
            ]
        }
    )
    with capture_assertion_returns(world, [assertion]) as events:
        assert not AssertionRegistry.check(world, assertion)
    message = failure_trace(assertion, events[id(assertion)])
    assert '"Invoice": "GL-88210", "Amount": "11,340.5", "Notes": "REVIEW"' in message
    assert '"SP-0091" compared with "GL-88210": no match' in message
    assert '"AS-7744" compared with "GL-88210": no match' in message
    assert f"Rows: {row_count} items (contents not captured)" in message
    assert "not exhaustive" in message and "overall failure" in message
    assert "_contains_normalized" not in message and "```" not in message and "<Row>" not in message
    # The invoice actually exists in an earlier row, but those comparisons
    # were evicted. The summary must not claim that the invoice is absent.
    assert "missing" not in message and "absent" not in message


def test_failure_summary_without_captured_events_keeps_requirements_and_states_limit() -> None:
    message = failure_trace({"type": "gmail_message_sent_to", "to": "a@example.com", "scored": True}, [])
    assert 'To: "a@example.com"' in message and "not captured" in message
    assert "scored" not in message and "Observed comparisons" not in message


def test_failure_summary_keeps_observations_that_differ_from_same_named_requirement() -> None:
    message = failure_trace(
        {"type": "example", "value": "required"},
        [{"function": "example", "returned": False, "values": {"value": "observed"}}],
    )
    assert 'Value: "required"' in message and 'Value: "observed"' in message


def test_failure_summary_total_size_is_bounded_with_many_unicode_observations() -> None:
    message = failure_trace(
        {"type": "example"},
        [
            {
                "function": "example",
                "returned": False,
                "values": {f"value_{index}": "界" * 400 for index in range(20)},
            }
        ],
    )
    assert len(message.encode("utf-8")) <= 6_000 and "diagnostic truncated" in message
    assert "Partial diagnostic:" in message and "overall failure" in message
