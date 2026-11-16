"""Dataset, prompt and native PI integration regression tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from automationbench.schema.world import WorldState
from beaker import CaseResult
from fake_client import ScriptedEndpoint

from automationbench_skills.data import BENCHMARK_DOMAINS, load_samples, load_split, task_family
from automationbench_skills.evaluation.summary import format_summary, summarize
from automationbench_skills.prompts import load_system_prompt, task_clock, with_system_prompt
from automationbench_skills.runner import ModelSpec, _rollout_input, run_rollout_raw
from automationbench_skills.scenario_clock import DEFAULT_SCENARIO_TIME, scenario_time
from automationbench_skills.skills_tools import list_skills, read_skill, set_skills_dir


RECIPE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(RECIPE_ROOT / ".beaker"))
from beaker_integration import TaskRow, TaskSetup, _fixture_for, score_case  # noqa: E402
from case_fixture import PI_COMMIT, CaseFixture  # noqa: E402


class TestSplits:
    def test_filtered_split_shape(self) -> None:
        train, test = load_split("train"), load_split("test")
        assert len(train) == 584 and len(test) == 191
        assert not {s.task_name for s in train} & {s.task_name for s in test}
        from automationbench_skills.data.tasks import read_split_names

        excluded = read_split_names("excluded_cases")
        retained = {s.task_name for s in train + test}
        assert len(excluded) == len(set(excluded)) == 25
        assert not retained & set(excluded)
        assert retained | set(excluded) == {s.task_name for s in load_samples()}
        for split, samples in (("train", train), ("test", test)):
            assert [s.task_name for s in samples] == [name for name in read_split_names(split) if name not in excluded]
        expected = {
            "sales": (72, 24),
            "marketing": (73, 24),
            "operations": (72, 23),
            "support": (70, 21),
            "finance": (74, 25),
            "hr": (73, 24),
            "simple": (150, 50),
        }
        for domain in BENCHMARK_DOMAINS:
            assert (sum(s.domain == domain for s in train), sum(s.domain == domain for s in test)) == expected[domain]

    def test_split_regeneration_is_deterministic(self) -> None:
        from automationbench_skills.data.make_splits import make_splits
        from automationbench_skills.data.tasks import read_split_names

        train, test = make_splits()
        assert train == read_split_names("train")
        assert test == read_split_names("test")

    def test_task_names_globally_unique(self) -> None:
        samples = load_samples()
        assert len({s.task_name for s in samples}) == len(samples) == 800

    def test_task_family(self) -> None:
        assert task_family("sales.docusign_contract_send") == "docusign"


class TestSkillsTools:
    @staticmethod
    def _write(root: Path, skill_id: str, description: str, body: str) -> None:
        p = root / skill_id / "SKILL.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\nname: {skill_id.split('/')[-1]}\ndescription: {description}\n---\n{body}\n")

    def test_nested_discovery_and_live_reload(self, tmp_path: Path) -> None:
        set_skills_dir(tmp_path)
        assert list_skills() == "No skills available."
        self._write(tmp_path, "apps/gmail", "Gmail procedures", "body text")
        self._write(tmp_path, "domains/finance", "Finance playbooks", "finance body")
        listing = list_skills()
        assert "apps/gmail: Gmail procedures" in listing
        assert "domains/finance: Finance playbooks" in listing
        assert "body text" in read_skill("apps/gmail")
        assert "finance body" in read_skill("domains/finance")
        self._write(tmp_path, "apps/gmail", "Changed", "new body")
        assert "apps/gmail: Changed" in list_skills()
        assert "new body" in read_skill("apps/gmail")

    def test_read_skill_missing(self, tmp_path: Path) -> None:
        set_skills_dir(tmp_path)
        self._write(tmp_path, "apps/gmail", "Gmail", "body")
        message = read_skill("nope")
        assert "unknown skill" in message.lower() and "apps/gmail" in message

    def test_shipped_seed_stubs(self) -> None:
        shipped = RECIPE_ROOT / "skills"
        set_skills_dir(shipped)
        listing = list_skills()
        for domain in BENCHMARK_DOMAINS:
            if domain != "simple":
                assert f"domains/{domain}: " in listing
        for app in ["gmail", "google_sheets", "google_drive", "slack", "salesforce"]:
            assert f"apps/{app}: " in listing
        assert "unknown skill" not in read_skill("apps/gmail").lower()


class TestPrompts:
    def test_file_replaces_the_system_message_and_keeps_the_task(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        out = with_system_prompt(prompt, "OURS")
        assert out == [{"role": "system", "content": "OURS"}, prompt[1]]
        assert prompt[0]["content"] == "BENCHMARK PROMPT"
        assert with_system_prompt(prompt, None) is prompt
        assert with_system_prompt(prompt, "") is prompt
        assert with_system_prompt("plain", "OURS") == "plain"
        assert with_system_prompt([prompt[1]], "OURS") == [{"role": "system", "content": "OURS"}, prompt[1]]

    def test_clock_appends_to_the_system_message(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        out = with_system_prompt(prompt, "OURS", clock="2026-03-10T09:00:00Z")
        assert out[0] == {"role": "system", "content": "OURS\n\nCurrent date and time: 2026-03-10T09:00:00Z"}
        assert out[1:] == prompt[1:]

    def test_clock_without_a_system_prompt_keeps_the_rows_system_message(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        out = with_system_prompt(prompt, None, clock="2026-03-10T09:00:00Z")
        assert out[0] == {
            "role": "system",
            "content": "BENCHMARK PROMPT\n\nCurrent date and time: 2026-03-10T09:00:00Z",
        }
        assert out[1:] == prompt[1:]
        assert prompt[0]["content"] == "BENCHMARK PROMPT"
        out = with_system_prompt([prompt[1]], None, clock="2026-03-10T09:00:00Z")
        assert out[0] == {"role": "system", "content": "Current date and time: 2026-03-10T09:00:00Z"}

    def test_no_clock_is_unchanged_behaviour(self) -> None:
        prompt = [{"role": "system", "content": "BENCHMARK PROMPT"}, {"role": "user", "content": "do the task"}]
        assert with_system_prompt(prompt, "OURS", clock=None) == [{"role": "system", "content": "OURS"}, prompt[1]]
        assert with_system_prompt(prompt, None, clock=None) is prompt
        assert with_system_prompt(prompt, "OURS", clock="") == [{"role": "system", "content": "OURS"}, prompt[1]]

    def test_task_clock_reads_dict_or_json_initial_state(self) -> None:
        info = {"initial_state": {"meta": {"current_time": "2026-03-10T09:00:00Z"}}}
        assert task_clock(info) == "2026-03-10T09:00:00Z"
        assert task_clock({"initial_state": json.dumps(info["initial_state"])}) == "2026-03-10T09:00:00Z"
        assert task_clock({}) is None
        assert task_clock({"initial_state": {"meta": {}}}) is None
        assert task_clock({"initial_state": {}}) is None
        assert task_clock({"initial_state": "not json"}) is None

    def test_load_reads_live_and_tolerates_absence(self, tmp_path: Path) -> None:
        assert load_system_prompt(None) is None
        assert load_system_prompt(tmp_path) is None
        (tmp_path / "system.md").write_text("  \n")
        assert load_system_prompt(tmp_path) is None
        (tmp_path / "system.md").write_text("first\n")
        assert load_system_prompt(tmp_path) == "first"
        (tmp_path / "system.md").write_text("second\n")
        assert load_system_prompt(tmp_path) == "second"
        assert load_system_prompt(tmp_path, skills=False) is None
        (tmp_path / "system_no_skills.md").write_text("plain\n")
        assert load_system_prompt(tmp_path, skills=False) == "plain"
        assert load_system_prompt(tmp_path) == "second"

    def test_shipped_seeds_start_with_the_benchmark_prompt_verbatim(self) -> None:
        upstream = {s.prompt[0]["content"] for s in load_split("train") + load_split("test")}
        assert len(upstream) == 1
        benchmark = upstream.pop().strip()
        assert load_system_prompt(RECIPE_ROOT / "prompts", skills=False) == benchmark
        text = load_system_prompt(RECIPE_ROOT / "prompts")
        assert text is not None
        assert text.startswith(benchmark)
        for domain in BENCHMARK_DOMAINS:
            assert domain in text
        assert "list_skills" in text and "read_skill" in text


class TestSummary:
    def test_summarize_and_format(self) -> None:
        rows = [
            {
                "domain": "sales",
                "task_completed_correctly": 1.0,
                "partial_credit": 1.0,
                "latency_s": 2.0,
                "cost_usd": 0.0123,
            },
            {
                "domain": "sales",
                "task_completed_correctly": 0.0,
                "partial_credit": 0.5,
                "latency_s": 4.0,
                "cost_usd": 0.0456,
            },
            {"domain": "hr", "task_completed_correctly": 0.0, "partial_credit": 0.0},
        ]
        summary = summarize(rows)
        assert summary["domains"]["sales"] == {
            "tasks": 2,
            "pass_rate": 0.5,
            "partial_credit": 0.75,
            "avg_latency_s": 3.0,
            "avg_cost_usd": 0.02895,
        }
        assert summary["domains"]["hr"]["avg_latency_s"] is None
        assert summary["domains"]["hr"]["avg_cost_usd"] is None
        assert summary["overall"]["tasks"] == 3
        text = format_summary(summary)
        assert "overall" in text and "sales" in text
        assert "3.0" in text and "0.0290" in text and "-" in text


@pytest.fixture
def row():
    sample = copy.deepcopy(next(s for s in load_samples() if s.task_name == "simple.email_sf_contact_phone_update"))
    task = {"example_id": sample.index, "prompt": sample.prompt, "answer": sample.answer, "info": sample.info}
    fixture = {
        "task": task,
        "scenario": {
            "source": "pi",
            "fallback_time": DEFAULT_SCENARIO_TIME,
            "tool_fault": None,
            "trace_capture": "complete",
        },
    }
    return {
        "id": "frozen-phone",
        "input": {"task_name": sample.task_name},
        "expected": {"assertions": copy.deepcopy(sample.info["assertions"])},
        "metadata": {"benchmark_commit": PI_COMMIT, **hashes(fixture)},
        "runtime_fixture": fixture,
    }


def hashes(fixture: dict) -> dict:
    return {
        f"{name}_sha256": hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        for name, value in fixture.items()
    }


def test_all_pi_tasks_preserve_their_metadata_and_hashes(row):
    from automationbench_skills.data.tasks import load_samples

    samples = load_samples()
    assert len(samples) == 800
    for sample in samples:
        raw = {
            "task": {
                "example_id": sample.index,
                "prompt": sample.prompt,
                "answer": sample.answer,
                "info": sample.info,
            },
            "scenario": row["runtime_fixture"]["scenario"],
        }
        fixture = CaseFixture.model_validate(raw)
        assert fixture.model_dump(mode="json") == raw, sample.task_name
        fixture.validate_reference(
            sample.task_name,
            {"assertions": sample.info["assertions"]},
            {"benchmark_commit": PI_COMMIT, **hashes(raw)},
        )
        assert fixture.sample().info.get("meta") == sample.info.get("meta")


def test_task_metadata_is_covered_by_the_hash(row):
    row["runtime_fixture"]["task"]["info"]["meta"] = {"current_time": "2026-02-10T09:00:00"}
    row["metadata"].update(hashes(row["runtime_fixture"]))
    TaskRow.model_validate(row)
    row["runtime_fixture"]["task"]["info"]["meta"]["current_time"] = "2026-02-11T09:00:00"
    with pytest.raises(ValueError, match="task_sha256"):
        TaskRow.model_validate(row)


def test_named_rows_reject_stale_assertions(row):
    row.pop("runtime_fixture")
    row["metadata"] = {"benchmark_commit": PI_COMMIT}
    row["expected"]["assertions"][0]["value"] = "stale@example.com"
    with pytest.raises(ValueError, match="expected.assertions"):
        TaskRow.model_validate(row)


async def test_named_quickstart_rows_resolve_to_frozen_cases():
    from upload_splits import TEST_PER_DOMAIN, TRAIN_PER_DOMAIN, _take_per_domain

    for split, count in (("train", TRAIN_PER_DOMAIN), ("test", TEST_PER_DOMAIN)):
        for row in _take_per_domain(split, count):
            case = await anext(TaskSetup().load_cases(TaskRow.model_validate(row), runtime=SimpleNamespace()))
            fixture = CaseFixture.model_validate(case.input["runtime_fixture"])
            assert fixture.task.info.task_name == row["input"]["task_name"]
            assert fixture.task.info.assertions == row["expected"]["assertions"]


@pytest.mark.parametrize("mutation", ["missing", "task", "assertions", "source"])
def test_rejects_missing_or_inconsistent_fixture(row, mutation):
    if mutation == "missing":
        row.pop("runtime_fixture")
    elif mutation == "task":
        row["runtime_fixture"]["task"]["info"]["initial_state"] = {}
    elif mutation == "assertions":
        row["expected"] = {"assertions": []}
    else:
        row["metadata"]["benchmark_commit"] = "unrecognized"
    with pytest.raises(ValueError):
        TaskRow.model_validate(row)


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
    async with ScriptedEndpoint([]) as endpoint:
        output = await run_rollout_raw(sample, model=ModelSpec(name="fake", base_url=endpoint.url))
    assert endpoint.requests[0]["messages"][0]["content"].endswith(f"Current date and time: {expected}")
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


@pytest.mark.parametrize("writes", ["changed", "reverted", "none"])
async def test_sheet_update_history_survives_hosted_scoring(row, tmp_path, writes):
    from automationbench.domains import get_combined_dataset

    for task in get_combined_dataset(["finance"]):
        info = json.loads(task["info"]) if isinstance(task["info"], str) else task["info"]
        if info["task_name"] == "finance.duplicate_payment_detection":
            task["info"] = info
            break
    row["runtime_fixture"]["task"] = task
    row["input"]["task_name"] = info["task_name"]
    row["expected"]["assertions"] = info["assertions"]
    row["metadata"].update(hashes(row["runtime_fixture"]))
    parsed = TaskRow.model_validate(row)
    case = await anext(TaskSetup().load_cases(parsed, runtime=SimpleNamespace()))
    sheets = WorldState(**parsed.runtime_fixture.initial_state()).google_sheets
    worksheet = next(ws for ws in sheets.worksheets if ws.id == "ws_jan_pmts")
    initial_rows = {r.row_id: r.cells for r in sheets.rows if r.worksheet_id == worksheet.id}
    headers = worksheet.headers or list(initial_rows[2])
    column = chr(ord("A") + headers.index("Flag"))
    calls = []
    row_ids = [] if writes == "none" else [2, 4, 5, 7]
    for row_id in row_ids:
        original = initial_rows[row_id]["Flag"]
        values = ["Potential duplicate"]
        if writes == "reverted":
            values.append(original)
        for value in values:
            calls.append(
                {
                    "name": "api_fetch",
                    "arguments": {
                        "endpoint": "sheets.spreadsheets.values.update",
                        "path": {"spreadsheetId": "ss_vendor_pmts", "range": f"'{worksheet.title}'!{column}{row_id}"},
                        "params": {"valueInputOption": "RAW"},
                        "body": {"values": [[value]]},
                    },
                }
            )
    async with ScriptedEndpoint([{"tool_calls": calls}]) as endpoint:
        live = await run_rollout_raw(
            parsed.runtime_fixture.sample(), model=ModelSpec(name="fake", base_url=endpoint.url)
        )
    result = CaseResult(
        output=None, context={"end_state": live["_end_state"], "grading_state": live["_grading_state"]}
    )
    # Exercise the hosted JSON boundary, not just in-process object passing.
    transported = CaseResult(output=result.output, context=json.loads(json.dumps(result.context)))
    score = await score_case(case=case, result=transported, case_files_dir=tmp_path)
    assert not live.get("error")
    expected = [True] * 6 if writes in {"changed", "reverted"} else [False, False, True, True, False, False]
    assert [a["passed"] for a in live["_assertion_results"][:6]] == expected
    assert [c.verdict == "pass" for c in score.checks[:6]] == expected
    assert score.field_scores == live["metrics"]
