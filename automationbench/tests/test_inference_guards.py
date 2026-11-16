"""Exercise the native runtime at its HTTP and hosted-scoring boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from beaker import RetryableCaseError
from beaker.tracing import current_trace, local_capture
from fake_client import ScriptedEndpoint

from automationbench_skills.data import load_samples
from automationbench_skills.runner import ModelSpec, run_rollout_raw


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
import beaker_integration as integration


PHONE_UPDATE = {
    "name": "api_fetch",
    "arguments": {
        "endpoint": "salesforce.sobjects.contact.update",
        "path": {"id": "003001"},
        "body": {"Phone": "+1-555-0101"},
    },
}


def sample():
    return next(s for s in load_samples() if s.task_name == "simple.email_sf_contact_phone_update")


async def case_for(s):
    row = integration.TaskRow.model_validate(
        {
            "id": s.task_name,
            "input": {"task_name": s.task_name},
            "expected": {"assertions": s.info["assertions"]},
            "metadata": {"benchmark_commit": integration.PI_COMMIT},
        }
    )
    return await anext(integration.TaskSetup().load_cases(row, runtime=SimpleNamespace()))


async def test_native_tools_score_and_trace_through_hosted_integration(monkeypatch, tmp_path):
    case = await case_for(sample())
    turns = [
        {"tool_calls": [{"name": "list_skills", "arguments": {}}]},
        {"tool_calls": [{"name": "read_skill", "arguments": {"skill_id": "apps/gmail"}}]},
        {"tool_calls": [PHONE_UPDATE]},
    ]
    async with ScriptedEndpoint(turns) as endpoint:
        model = ModelSpec(name="fake", base_url=endpoint.url, extra_body='{"max_tokens": 512}')
        monkeypatch.setattr(integration, "_model_for", lambda _: (model, "test-run-key"))
        with local_capture(tmp_path, case_id=case.id, candidate_id="native", strict_evidence=True) as capture:
            result = await integration.run_case(case_input=case.input, runtime=SimpleNamespace(trace=current_trace()))
        score = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    assert score.field_scores == {"partial_credit": 1.0, "task_completed_correctly": 1.0}
    assert result.context["end_state"]["salesforce"]["contacts"][0]["phone"] == "+1-555-0101"
    receipt = capture.receipt.to_dict()
    assert receipt["status"] == "complete"
    assert receipt["counts"]["model_calls"] == 4
    assert receipt["projection"]["tool_counts"] == {"list_skills": 1, "read_skill": 1, "api_fetch": 1}
    assert receipt["projection"]["token_usage"]["total_tokens"] == 60
    assert all(value == "Bearer test-run-key" for value in endpoint.authorizations)
    assert all(body.get("max_completion_tokens", body.get("max_tokens")) == 512 for body in endpoint.requests)
    assert "test-run-key" not in json.dumps(receipt)
    results = [m["content"] for m in endpoint.requests[-1]["messages"] if m["role"] == "tool"]
    assert "apps/gmail" in results[0]
    assert "Gmail" in results[1] and "unknown skill" not in results[1]


@pytest.mark.parametrize("failure", ["deadline", "provider"])
async def test_native_failure_preserves_completed_writes(failure, monkeypatch, tmp_path):
    turns = [
        {"tool_calls": [PHONE_UPDATE]},
        {"delay": 15, "content": "Done"} if failure == "deadline" else {"status": 401},
    ]
    async with ScriptedEndpoint(turns) as endpoint:
        model = ModelSpec(name="fake", base_url=endpoint.url)
        with local_capture(tmp_path, case_id="failure", candidate_id="native", strict_evidence=False):
            result = await run_rollout_raw(sample(), model=model, timeout=5 if failure == "deadline" else 20)
        assert result["reward"] == 1.0
        assert result["_end_state"]["salesforce"]["contacts"][0]["phone"] == "+1-555-0101"
        if failure == "deadline":
            assert result["stop_condition"] == "timeout_reached"
        else:
            assert result["error"].type == "ProviderError"
            # The saved partial state is useful evidence, but provider failure must
            # ask Beaker to retry rather than count as a completed evaluation.
            from unittest.mock import AsyncMock

            monkeypatch.setattr(integration, "run_rollout_raw", AsyncMock(return_value=result))
            monkeypatch.setattr(integration, "_model_for", lambda _: (model, None))
            case = await case_for(sample())
            with local_capture(tmp_path / "hosted", case_id=case.id, candidate_id="native"):
                with pytest.raises(RetryableCaseError):
                    await integration.run_case(case_input=case.input, runtime=SimpleNamespace(trace=current_trace()))


@pytest.mark.parametrize("api", ["anthropic", "responses", "gemini_interactions"])
def test_unsupported_native_apis_fail_clearly(api):
    with pytest.raises(ValueError, match="Chat Completions"):
        ModelSpec(api=api)


@pytest.mark.parametrize("budget", [0, -1, True, "1024"])
def test_invalid_reasoning_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        ModelSpec(extra_body=json.dumps({"reasoning": {"max_tokens": budget}})).sampling_args()


def test_openrouter_configuration():
    model = ModelSpec(name="vendor/model", reasoning_effort="high", reasoning_enabled=True)
    assert model.effective_base_url() == "https://openrouter.ai/api/v1"
    assert model.effective_api_key_var() == "OPENROUTER_API_KEY"
    assert model.sampling_args()["extra_body"]["reasoning"] == {"effort": "high", "enabled": True}
