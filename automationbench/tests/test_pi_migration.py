"""Migration checks against the real PI tools, published rows and scoring."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from automationbench.rubric import partial_credit
from automationbench.schema.world import WorldState
from beaker import Case, CaseResult
from fake_client import ScriptedClient

from automationbench_skills.data import load_samples, load_split
from automationbench_skills.data.tasks import read_split_names
from automationbench_skills.model_protocol import SystemMessage, ToolMessage, UserMessage
from automationbench_skills.runner import _rollout_input, get_env
from automationbench_skills.vendored.clients import (
    GeminiInteractionsClient,
    OpenAIResponsesClient,
    StreamingAnthropicClient,
)


ROOT = Path(__file__).parents[1]
PI_COMMIT = "858718b19e5d66c74451432783d34ee64a550a15"


def phone_sample():
    return next(s for s in load_samples() if s.task_name == "simple.email_sf_contact_phone_update")


@pytest.mark.parametrize("mode", ["zapier", "api", "limited_zapier", "harness"])
async def test_pi_tools_change_the_world_and_earn_native_credit(mode):
    sample = phone_sample()
    values = {"id": "003001", "phone": "+1-555-0101"}
    call = {"name": "salesforce_contact_update", "arguments": values}
    if mode == "zapier":
        call = {"name": "execute_tool", "arguments": {"tool_name": call["name"], "arguments": json.dumps(values)}}
    elif mode == "api":
        call = {
            "name": "api_fetch",
            "arguments": {
                "endpoint": "salesforce.sobjects.contact.update",
                "path": {"id": "003001"},
                "body": {"Phone": "+1-555-0101"},
            },
        }
    env = get_env(toolset=mode, skills=False)
    output = await env.run_rollout(_rollout_input(sample), ScriptedClient([{"tool_calls": [call]}]), "fake", {})
    assert output.get("error") is None
    assert output["_assertion_results"][0]["passed"] is True
    assert output["metrics"] == {"partial_credit": 1.0, "task_completed_correctly": 1.0}
    assert output["_end_state"]["salesforce"]["contacts"][0]["phone"] == "+1-555-0101"
    assert sample.info["initial_state"]["salesforce"]["contacts"][0]["phone"] == "+1-555-0000"
    assert (
        partial_credit(
            {
                "info": sample.info,
                "initial_state": sample.info["initial_state"],
                "world": WorldState(**output["_end_state"]),
            }
        )
        == output["reward"]
    )


async def test_world_and_clock_are_isolated_in_concurrent_rollouts():
    sample = phone_sample()
    env = get_env(skills=False)

    def client(phone):
        return ScriptedClient(
            [
                {
                    "tool_calls": [
                        {
                            "name": "execute_tool",
                            "arguments": {
                                "tool_name": "salesforce_contact_update",
                                "arguments": json.dumps({"id": "003001", "phone": phone}),
                            },
                        }
                    ]
                }
            ]
        )

    good, bad = await asyncio.gather(
        env.run_rollout(_rollout_input(sample), client("+1-555-0101"), "fake", {}),
        env.run_rollout(_rollout_input(sample), client("wrong"), "fake", {}),
    )
    assert (good["reward"], bad["reward"]) == (1.0, 0.0)
    assert good["_end_state"]["meta"]["current_time"] == bad["_end_state"]["meta"]["current_time"]


async def test_api_schema_preserves_pi_unions_and_invalid_call_leaves_world_unchanged():
    sample = phone_sample()
    client = ScriptedClient(
        [
            {
                "tool_calls": [
                    {
                        "name": "api_fetch",
                        "arguments": {
                            "endpoint": "salesforce.sobjects.contact.update",
                            "path": {"wrong_id": "003001"},
                            "body": {"Phone": "wrong"},
                        },
                    }
                ]
            }
        ]
    )
    output = await get_env(toolset="api", skills=False).run_rollout(_rollout_input(sample), client, "fake", {})
    schema = next(t["parameters"] for t in client.calls[0]["tools"] if t["name"] == "api_fetch")
    assert schema["allOf"][0]["oneOf"]
    assert output["reward"] == 0
    assert output["_end_state"]["salesforce"]["contacts"][0]["phone"] == "+1-555-0000"
    assert "error" in output["completion"][1].content.lower()


def test_published_rows_cover_every_split_and_exclusions_match_documentation():
    train, test = set(read_split_names("train")), set(read_split_names("test"))
    assert train | test == {s.task_name for s in load_samples()}
    assert not train & test
    excluded = set(read_split_names("excluded_cases"))
    text = (ROOT / "src/automationbench_skills/splits/EXCLUSIONS.md").read_text()
    import re

    assert set(re.findall(r"^\| \[([\w.]+)\]", text, re.M)) == excluded
    assert len(excluded) == 25
    assert len(load_split("train")) + len(load_split("test")) == 775


async def test_upload_rows_carry_pi_assertions_and_old_dataset_is_rejected():
    sys.path.insert(0, str(ROOT / ".beaker"))
    from beaker_integration import TaskRow, TaskSetup
    from upload_splits import _take_per_domain

    rows = _take_per_domain("train", 6)
    assert len(rows) == 42
    assert {r["metadata"]["benchmark_commit"] for r in rows} == {PI_COMMIT}
    samples = {s.task_name: s for s in load_samples()}
    assert all(row["expected"]["assertions"] == samples[row["id"]].info["assertions"] for row in rows)
    setup = TaskSetup()
    assert len([case async for case in setup.load_cases(TaskRow.model_validate(rows[0]), runtime=None)]) == 1
    rows[0]["metadata"].pop("benchmark_commit")
    with pytest.raises(ValueError, match="benchmark commit"):
        _ = [case async for case in setup.load_cases(TaskRow.model_validate(rows[0]), runtime=None)]


async def test_hosted_score_reports_pi_negative_credit_gating(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / ".beaker"))
    import beaker_integration as integration

    assertions = [
        {"type": "gmail_message_sent_to", "to": "target@example.com"},
        {"type": "gmail_message_not_sent_to", "to": "excluded@example.com"},
    ]
    monkeypatch.setattr(integration, "_authored_initial_states", lambda: {"sample": {}})
    score = await integration.score_case(
        case=Case(id="sample", input={"task_name": "sample"}, expected={"assertions": assertions}),
        result=CaseResult(output=None, context={"end_state": WorldState().model_dump(mode="json")}),
        case_files_dir=tmp_path,
    )
    assert score.field_scores["partial_credit"] == 0
    assert not score.checks[0].informational
    assert score.checks[1].informational
    assert "positive requirements" in score.checks[1].message


async def test_responses_adapter_replays_reasoning_and_function_call_items():
    client = OpenAIResponsesClient(object())
    items = [
        SimpleNamespace(type="reasoning", id="rs", summary=[], encrypted_content="opaque"),
        SimpleNamespace(
            type="function_call", id="fc", call_id="call1", name="read_skill", arguments='{"skill_id":"apps/gmail"}'
        ),
    ]
    response = await client.from_native_response(
        SimpleNamespace(id="r", created_at=1, model="test", output=items, usage=None)
    )
    prompt, extra = await client.to_native_prompt(
        [
            SystemMessage(content="SYS"),
            UserMessage(content="task"),
            response.message,
            ToolMessage(tool_call_id="call1", content="guide"),
        ]
    )
    assert extra["instructions"] == "SYS"
    assert prompt[1]["encrypted_content"] == "opaque"
    assert prompt[2]["call_id"] == "call1"
    assert prompt[3] == {"type": "function_call_output", "call_id": "call1", "output": "guide"}


async def test_anthropic_adapter_preserves_thinking_signature_across_tool_turn():
    from anthropic.types import Message

    native = Message.model_validate(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {"type": "thinking", "thinking": "reason", "signature": "sig"},
                {"type": "tool_use", "id": "call1", "name": "list_skills", "input": {}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )
    client = StreamingAnthropicClient(object())
    response = await client.from_native_response(native)
    prompt, _ = await client.to_native_prompt([response.message, ToolMessage(tool_call_id="call1", content="guide")])
    assert prompt[0]["content"][0]["signature"] == "sig"
    assert prompt[1]["content"][0]["tool_use_id"] == "call1"


async def test_gemini_adapter_keeps_function_result_identity_and_usage():
    client = GeminiInteractionsClient(object())
    response = await client.from_native_response(
        {
            "id": "i",
            "model": "test",
            "created": 1,
            "steps": [{"type": "function_call", "id": "call1", "name": "list_skills", "arguments": {}}],
            "usage": {"total_input_tokens": 2, "total_output_tokens": 3, "total_thought_tokens": 4},
        }
    )
    assert response.usage.completion_tokens == 7
    prompt, _ = await client.to_native_prompt([response.message, ToolMessage(tool_call_id="call1", content="guide")])
    assert prompt[-1]["name"] == "list_skills"


def test_original_600_tasks_keep_their_split_membership_and_order():
    import hashlib

    # Digests of the original source splits at cookbook a3c0b83fbc5e.
    expected = {
        "train": "941e38c1e7deecdf10760304f04d3c2fbb86a8d9d20b6078806d46e42c9bbf66",
        "test": "4f217906559ea7bf57d54e2094c9abb0042f1bbcae54a2bcff67a8e38c0c8b11",
    }
    for split, digest in expected.items():
        names = [name for name in read_split_names(split) if not name.startswith("simple.")]
        assert hashlib.sha256("\n".join(names).encode()).hexdigest() == digest
