"""Regression coverage for condition Checks and shared observed OUTPUT."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from automationbench.schema.world import WorldState
from beaker import Case, CaseResult


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
import beaker_integration as integration  # noqa: E402
from world_diff import OUTPUT_MAX_BYTES, ServiceDiffs  # noqa: E402


def test_failed_email_check_has_scoped_message_without_a_value_pair() -> None:
    world = WorldState(
        gmail={
            "messages": [
                {"to": ["target@example.com"], "label_ids": ["SENT"], "body_plain": "Wrong total"},
                {"to": ["other@example.com"], "label_ids": ["SENT"], "body_plain": "Unrelated email"},
                {"to": ["target@example.com"], "label_ids": ["INBOX"], "body_plain": "Unsent email"},
            ]
        }
    )
    check = integration._check(
        {
            "type": "gmail_message_sent_to_with_body_contains",
            "passed": False,
            "params": {"to": "target@example.com", "body_contains": ["19,500"]},
        },
        error=None,
        entities={},
        world=world,
    )
    assert check.expected is None and check.predicted is None and check.verdict == "fail"
    assert "Wrong total" in check.message
    assert "Unrelated email" not in check.message and "Unsent email" not in check.message
    assert "19,500" in check.description
    missing = integration._gmail_failure_message(world, {"to": "missing@example.com", "body_contains": ["total"]})
    assert "target@example.com" in missing and "other@example.com" in missing
    assert "Wrong total" not in missing and "Unrelated email" not in missing


@pytest.mark.parametrize(("passed", "excluded"), [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize(
    ("assertion_type", "params"),
    [
        ("salesforce_campaign_member_exists", {"contact_id": "c1", "campaign_id": "p1"}),
        ("gmail_draft_body_contains", {"to": "c1", "body_contains": "p1"}),
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
        assert "see OUTPUT" in check.message


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
