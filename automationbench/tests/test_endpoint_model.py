"""Agent-configured endpoint models: settings, precedence, and the requests they send."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from beaker.tracing import current_trace, local_capture
from fake_client import ScriptedEndpoint

from automationbench_skills.data import load_samples


sys.path.insert(0, str(Path(__file__).parents[1] / ".beaker"))
import beaker_integration as integration


SAMPLING = {
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": 8192,
    "chat_template_kwargs": {"enable_thinking": False},
}
PHONE_UPDATE = {
    "name": "api_fetch",
    "arguments": {
        "endpoint": "salesforce.sobjects.contact.update",
        "path": {"id": "003001"},
        "body": {"Phone": "+1-555-0101"},
    },
}


@pytest.fixture(autouse=True)
def no_endpoint(monkeypatch):
    for name in (
        integration.MODEL_ENV,
        integration.BASE_URL_ENV,
        integration.API_KEY_VAR_ENV,
        integration.EXTRA_BODY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def set_endpoint(monkeypatch, base_url="http://endpoint.test/v1", extra_body=SAMPLING):
    monkeypatch.setenv(integration.MODEL_ENV, "qwen3.5-9b")
    monkeypatch.setenv(integration.BASE_URL_ENV, base_url)
    monkeypatch.setenv(integration.API_KEY_VAR_ENV, "AB_ENDPOINT_KEY")
    monkeypatch.setenv("AB_ENDPOINT_KEY", "endpoint-secret")
    if extra_body is not None:
        monkeypatch.setenv(integration.EXTRA_BODY_ENV, json.dumps(extra_body))


def test_without_endpoint_settings_the_default_model_is_used():
    model, api_key = integration._model_for(SimpleNamespace(model=None))
    assert model == integration.default_model_spec()
    assert api_key is None


def test_endpoint_settings_select_the_endpoint_with_its_sampling(monkeypatch):
    set_endpoint(monkeypatch)
    model, api_key = integration._model_for(SimpleNamespace(model=None))
    assert (model.name, model.base_url, api_key) == ("qwen3.5-9b", "http://endpoint.test/v1", "endpoint-secret")
    args = model.sampling_args()
    assert args["max_completion_tokens"] == 8192
    assert args["extra_body"] == {k: v for k, v in SAMPLING.items() if k != "max_tokens"}
    assert "reasoning_effort" not in args


def test_a_run_model_choice_wins_over_the_endpoint(monkeypatch):
    set_endpoint(monkeypatch)
    target = SimpleNamespace(model="gateway-model", base_url="http://gateway.test/v1", api_key="run-token")
    monkeypatch.setattr(integration, "inference_target", lambda runtime: target)
    model, api_key = integration._model_for(SimpleNamespace(model="gateway-model"))
    assert (model.name, model.base_url, api_key) == ("gateway-model", "http://gateway.test/v1", "run-token")


@pytest.mark.parametrize(
    ("unset", "message"),
    [(integration.BASE_URL_ENV, "needs"), (integration.API_KEY_VAR_ENV, "needs"), ("AB_ENDPOINT_KEY", "Set")],
)
def test_incomplete_endpoint_settings_fail(monkeypatch, unset, message):
    set_endpoint(monkeypatch)
    monkeypatch.delenv(unset)
    with pytest.raises(ValueError, match=message):
        integration.endpoint_model_spec()


@pytest.mark.parametrize("extra_body", ["not json", "[1, 2]"])
def test_extra_body_must_be_a_json_object(monkeypatch, extra_body):
    set_endpoint(monkeypatch, extra_body=None)
    monkeypatch.setenv(integration.EXTRA_BODY_ENV, extra_body)
    with pytest.raises(ValueError, match="JSON object"):
        integration.endpoint_model_spec()


async def test_endpoint_requests_carry_the_key_and_sampling(monkeypatch, tmp_path):
    sample = next(s for s in load_samples() if s.task_name == "simple.email_sf_contact_phone_update")
    row = integration.TaskRow.model_validate(
        {
            "id": sample.task_name,
            "input": {"task_name": sample.task_name},
            "expected": {"assertions": sample.info["assertions"]},
            "metadata": {"benchmark_commit": integration.PI_COMMIT},
        }
    )
    async with ScriptedEndpoint([{"tool_calls": [PHONE_UPDATE]}]) as endpoint:
        set_endpoint(monkeypatch, base_url=endpoint.url)
        case = await anext(integration.TaskSetup().load_cases(row, runtime=SimpleNamespace()))
        with local_capture(tmp_path, case_id=case.id, candidate_id="endpoint", strict_evidence=True):
            result = await integration.run_case(
                case_input=case.input, runtime=SimpleNamespace(trace=current_trace(), model=None)
            )
        score = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    assert score.field_scores == {"partial_credit": 1.0, "task_completed_correctly": 1.0}
    assert endpoint.authorizations and all(a == "Bearer endpoint-secret" for a in endpoint.authorizations)
    for body in endpoint.requests:
        assert body["model"] == "qwen3.5-9b"
        assert body["temperature"] == 0.8 and body["top_p"] == 1.0 and body["top_k"] == 0
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body.get("max_completion_tokens", body.get("max_tokens")) == 8192
