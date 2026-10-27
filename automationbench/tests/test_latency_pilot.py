"""Offline integrity checks for the frozen H20 experiment."""

import importlib.util
import shutil
from pathlib import Path

import pytest


PILOT = Path(__file__).parents[1] / ".beaker" / "latency-pilot"
loader = importlib.util.spec_from_file_location("verify_latency_pilot", PILOT / "verify_pilot.py")
assert loader and loader.loader
verifier = importlib.util.module_from_spec(loader)
loader.loader.exec_module(verifier)


def test_pilot_preserves_h20_and_the_frozen_split():
    result = verifier.verify()
    assert result["candidate_files"] == 36
    assert result["train_cases"] == 36
    assert result["test_cases"] == 18
    assert result["max_concurrency"] == 32


@pytest.mark.parametrize("relative", ["project/prompts/system.md", "project/uv.lock", "dataset/test.jsonl"])
def test_pilot_rejects_changed_candidate_dependencies_or_data(tmp_path, relative):
    copy = tmp_path / "pilot"
    shutil.copytree(PILOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    path = copy / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        verifier.verify(copy)


def test_trusted_pilot_client_rejects_model_and_reasoning_changes():
    import asyncio
    import sys

    trusted = PILOT / "project" / ".beaker"
    spec = importlib.util.spec_from_file_location("h20_pilot_policy", trusted / "beaker_integration.py")
    assert spec and spec.loader
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    sys.path.insert(0, str(trusted))
    try:
        spec.loader.exec_module(policy)
    finally:
        sys.path.pop(0)
    assert policy.FIELD_WEIGHTS == {"partial_credit": 0.8, "task_completed_correctly": 0.2}
    client = object.__new__(policy._TracedChatCompletionsClient)
    with pytest.raises(RuntimeError, match="pins model"):
        asyncio.run(client.get_native_response([], "another-model", {"reasoning_effort": "medium"}))
    with pytest.raises(RuntimeError, match="pins medium reasoning"):
        asyncio.run(client.get_native_response([], "gpt-5.6-luna", {"reasoning_effort": "low"}))
