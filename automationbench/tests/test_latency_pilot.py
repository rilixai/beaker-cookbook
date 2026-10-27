"""Offline integrity checks for the frozen H20 experiment."""

import hashlib
import importlib.util
import json
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
    assert result["dataset_rows_verified"] is False
    assert not (PILOT / "dataset").exists()


@pytest.mark.parametrize(
    "relative", ["project/prompts/system.md", "project/uv.lock", "project/.beaker/beaker_integration.py"]
)
def test_pilot_rejects_changed_candidate_dependencies_or_policy(tmp_path, relative):
    copy = tmp_path / "pilot"
    shutil.copytree(PILOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    path = copy / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        verifier.verify(copy)


def test_external_source_data_is_verified_without_writing_pilot_rows(tmp_path):
    copy = tmp_path / "pilot"
    shutil.copytree(PILOT, copy, ignore=shutil.ignore_patterns("__pycache__"))
    source = tmp_path / "external-source"
    source.mkdir()
    provenance_path = copy / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    # Synthetic metadata-only rows exercise verification without embedding real tasks or assertions.
    for split in ("train", "test"):
        manifest = provenance["dataset"][split]
        rows = [
            {"id": case_id, "metadata": {"domain": f"domain-{i // manifest['per_domain']}", "source_split": split}}
            for i, case_id in enumerate(manifest["case_ids"])
        ]
        data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
        (source / f"{split}.jsonl").write_bytes(data)
        manifest.update(
            source_cases=len(rows),
            source_sha256=hashlib.sha256(data).hexdigest(),
            pilot_sha256=hashlib.sha256(data).hexdigest(),
        )
    provenance_path.write_text(json.dumps(provenance))
    assert verifier.verify(copy, source_dataset_dir=source)["dataset_rows_verified"] is True
    assert not (copy / "dataset").exists()
    with (source / "test.jsonl").open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValueError, match="Source test dataset changed"):
        verifier.verify(copy, source_dataset_dir=source)


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
