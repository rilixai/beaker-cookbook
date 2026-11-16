"""Authored dates agree across prompts, execution worlds and hosted scoring."""

from __future__ import annotations

import copy
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fake_client import ScriptedClient

from automationbench_skills.data.tasks import PI_COMMIT, load_samples
from automationbench_skills.runner import _rollout_input, get_env
from automationbench_skills.scenario_clock import DEFAULT_SCENARIO_TIME, scenario_time


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
from beaker_integration import TaskRow, TaskSetup, _fixture_for  # noqa: E402


async def standard_fixture(sample):
    row = TaskRow.model_validate(
        {
            "id": sample.task_name,
            "input": {"task_name": sample.task_name},
            "expected": {"assertions": sample.info["assertions"]},
            "metadata": {"benchmark_commit": PI_COMMIT},
        }
    )
    case = await anext(TaskSetup().load_cases(row, runtime=None))
    return _fixture_for(case.input)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Today is 2026-02-09.", "2026-02-09T12:00:00+00:00"),
        ("Today is Monday, 2026-02-09.", "2026-02-09T12:00:00+00:00"),
        ("Today is January 27, 2026.", "2026-01-27T12:00:00+00:00"),
        ("Today is Feb 9, 2026.", "2026-02-09T12:00:00+00:00"),
        ("Today is October 9, 2026.", "2026-10-09T12:00:00+00:00"),
        ("Use today's date (2026-01-21).", "2026-01-21T12:00:00+00:00"),
        ("Due within 14 days from today (2026-02-09).", "2026-02-09T12:00:00+00:00"),
        ("Current time is 2026-02-09T14:00:00Z.", "2026-02-09T14:00:00+00:00"),
        ("Today is 2026-02-09. Current time is 2026-02-09T14:00:00Z.", "2026-02-09T14:00:00+00:00"),
        ("The event is February 28, 2026. Schedule it for tomorrow.", DEFAULT_SCENARIO_TIME),
        ("Send the report today. Its deadline is 2026-02-28.", DEFAULT_SCENARIO_TIME),
    ],
)
def test_explicit_current_dates_and_deterministic_fallback(text, expected):
    assert scenario_time([{"role": "user", "content": text}], {}) == expected


def test_authored_world_clock_takes_precedence_and_inputs_are_unchanged():
    prompt = [{"role": "user", "content": "Today is 2026-02-09."}]
    initial = {"meta": {"current_time": "2026-02-09T08:30:00-05:00"}}
    before = copy.deepcopy((prompt, initial))
    assert scenario_time(prompt, initial) == "2026-02-09T08:30:00-05:00"
    assert (prompt, initial) == before


def test_conflicting_current_dates_fail_instead_of_selecting_one():
    with pytest.raises(ValueError, match="conflicting"):
        scenario_time([{"role": "user", "content": "Today is 2026-02-09. Today is 2026-02-10."}], {})


async def test_all_published_tasks_have_the_same_cli_and_hosted_clock():
    for sample in load_samples():
        fixture = await standard_fixture(sample)
        assert sample.info["initial_state"] == fixture.initial_state(), sample.task_name
        assert _rollout_input(sample)["prompt"] == _rollout_input(fixture.sample())["prompt"], sample.task_name


@pytest.mark.parametrize(
    "name",
    [
        "operations.generator_load_testing_schedule",
        "operations.insurance_cert_validation",
        "operations.sensor_monitoring_alert",
    ],
)
async def test_authored_february_ninth_is_used_in_prompt_and_execution(name):
    sample = next(sample for sample in load_samples() if sample.task_name == name)
    expected = "2026-02-09T12:00:00+00:00"
    prompt_before = copy.deepcopy(sample.prompt)
    assert scenario_time(sample.prompt, {}) == expected
    assert sample.prompt == prompt_before
    fixture = await standard_fixture(sample)
    assert fixture.sample().prompt == prompt_before
    assert fixture.initial_state()["meta"]["current_time"] == expected
    client = ScriptedClient()
    output = await get_env(toolset="api", skills=False).run_rollout(_rollout_input(sample), client, "fake", {})
    assert client.calls[0]["prompt"][0].content.endswith(f"Current date and time: {expected}")
    assert datetime.fromisoformat(output["_end_state"]["meta"]["current_time"]) == datetime.fromisoformat(expected)


async def test_frozen_fixture_keeps_its_saved_fallback_and_hashes():
    sample = next(
        sample for sample in load_samples() if sample.task_name == "operations.generator_load_testing_schedule"
    )
    fixture = await standard_fixture(sample)
    fixture.scenario.fallback_time = "2026-02-20T12:00:00+00:00"
    raw = fixture.model_dump(mode="json")
    hashes = fixture.hashes()
    restored = _fixture_for({"runtime_fixture": raw})
    assert restored.initial_state()["meta"]["current_time"] == fixture.scenario.fallback_time
    assert restored.hashes() == hashes
    assert restored.model_dump(mode="json") == raw
