from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from backports import zstd
from beaker import Case, CaseResult
from beaker.tracing import current_trace
from typesafe_sdk import AsyncTypeSafeClient

from jev_banking77 import cli
from jev_banking77.agent import LABELS, MODEL, Prediction, classify, question
from jev_banking77.data import SPLIT_FILES, Expected, Message, Row, make_splits, source_file
from jev_banking77.evaluation import Outcome, summarize


ROOT = Path(__file__).resolve().parents[1]
TEXT = "I am still waiting on my card?"  # First real upstream training example.
INTENT = "card_arrival"


def distribution(label: str = INTENT) -> dict[str, float]:
    return {name: float(name == label) for name in LABELS}


def prediction(label: str = INTENT) -> Prediction:
    return Prediction(intent=label, probabilities=distribution(label), model=MODEL, input_tokens=100, latency_ms=25)


def client_for(received: list[dict[str, object]]) -> AsyncTypeSafeClient:
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["authorization"] == "Bearer offline-test-key"
        body = request.content
        if request.headers.get("content-encoding") == "zstd":
            body = zstd.decompress(body)
        received.append(json.loads(body))
        return httpx2.Response(
            200,
            json={
                "model": MODEL,
                "answers": {
                    "intent": {"type": "choice", "choice": INTENT, "confidence": 1.0, "probabilities": distribution()}
                },
                "usage": {"input_tokens": 100, "output_tokens": 2},
            },
        )

    return AsyncTypeSafeClient(api_key="offline-test-key", transport=httpx2.MockTransport(respond))


@pytest.mark.asyncio
async def test_real_sdk_sends_native_choice_and_reads_usage() -> None:
    received: list[dict[str, object]] = []
    async with client_for(received) as client:
        result = await classify(TEXT, client=client, routing_question=question(ROOT / "prompts"))
    assert result.intent == INTENT
    assert result.input_tokens == 100
    assert result.output_tokens == 2
    assert result.model == MODEL
    assert received[0]["state"] == TEXT
    assert received[0]["model"] == MODEL
    assert set(received[0]) == {"state", "questions", "model"}
    assert set(received[0]["questions"]["intent"]["criteria"]) == set(LABELS)  # type: ignore[index]


def test_candidate_cannot_remove_or_rename_intents(tmp_path: Path) -> None:
    (tmp_path / "instructions.md").write_text("Choose an intent.")
    (tmp_path / "criteria.json").write_text(json.dumps({"invented_intent": "A new label"}))
    with pytest.raises(ValueError, match="77 intent IDs"):
        question(tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_invalid_probabilities_are_rejected(bad: float) -> None:
    probabilities = distribution()
    probabilities[INTENT] = bad
    with pytest.raises(ValueError):
        Prediction(intent=INTENT, probabilities=probabilities, model=MODEL, latency_ms=1)


def fake_csv(split: str) -> bytes:
    """Synthetic markers test splitting only; never used as evaluation data."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["text", "category"])
    for label_index, label in enumerate(LABELS):
        count = (129 if label_index < 7 else 130) if split == "train" else 40
        for index in range(count):
            writer.writerow([f"synthetic-{split}-{label}-{index}", label])
    return stream.getvalue().encode()


def test_splits_are_reproducible_disjoint_and_keep_official_test_separate() -> None:
    train, test = fake_csv("train"), fake_csv("test")
    splits = make_splits(train, test)
    assert splits == make_splits(train, test)
    assert {key: len(rows) for key, rows in splits.items()} == {"train": 8463, "validation": 1540, "test": 3080}
    assert set(Counter(r.expected.intent for r in splits["validation"]).values()) == {20}
    sets = [{r.input.text for r in splits[name]} for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert all(row.id.startswith("banking77-test-") for row in splits["test"])
    assert SPLIT_FILES["test"].startswith("heldout/")
    assert SPLIT_FILES["validation"] == "optimization/test.jsonl"


def test_corrupt_cache_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        source_file("train.csv", tmp_path)


def test_metrics_include_request_failures_in_accuracy_and_f1() -> None:
    metrics = summarize(
        [
            Outcome(id="1", expected=INTENT, prediction=prediction()),
            Outcome(id="2", expected=INTENT, prediction=prediction("cash_withdrawal_charge")),
            Outcome(id="3", expected=INTENT, error="TimeoutError"),
        ]
    )
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["macro_f1_77_labels"] == pytest.approx(0.5 / 77)
    assert metrics["errors"] == 1
    assert metrics["calibration_cases"] == 2
    assert metrics["brier_score"] == 1
    assert metrics["ece_10_bins"] == 0.5
    assert metrics["estimated_cost_usd"] is None


def test_cost_requires_complete_usage() -> None:
    result = prediction()
    outcome = Outcome(id="1", expected=INTENT, prediction=result)
    assert summarize([outcome])["estimated_cost_usd"] == pytest.approx(0.0000042)
    result.input_tokens = None
    assert summarize([outcome])["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_cli_persists_errors_instead_of_silently_dropping_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = Row(id="1", input=Message(text=TEXT), expected=Expected(intent=INTENT))
    source = tmp_path / SPLIT_FILES["validation"]
    source.parent.mkdir()
    source.write_text(row.model_dump_json() + "\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test-key")

    async def fail(*args: object, **kwargs: object) -> Prediction:
        raise TimeoutError("request failed")

    monkeypatch.setattr(cli, "classify", fail)
    output = tmp_path / "result.jsonl"
    metrics = await cli.evaluate(
        argparse.Namespace(data=tmp_path, split="validation", limit=None, prompts=ROOT / "prompts", output=output)
    )
    assert metrics["accuracy"] == 0
    assert metrics["errors"] == 1
    assert json.loads(output.read_text())["error"] == "TimeoutError"
    assert output.with_suffix(".summary.json").exists()


@pytest.mark.asyncio
async def test_integration_uses_candidate_prompt_and_keeps_expected_label_out_of_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("banking_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    integration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integration)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "instructions.md").write_text("Candidate-specific routing instructions.")
    (prompts / "criteria.json").write_text((ROOT / "prompts/criteria.json").read_text())
    monkeypatch.chdir(tmp_path)
    received: list[dict[str, object]] = []
    monkeypatch.setattr(integration, "AsyncTypeSafeClient", lambda **kwargs: client_for(received))
    runtime = SimpleNamespace(model=None, trace=current_trace())
    result = await integration.run_case(case_input={"text": TEXT}, runtime=runtime)
    assert received[0]["state"] == TEXT
    assert "expected" not in received[0]
    assert received[0]["questions"]["intent"]["instructions"] == "Candidate-specific routing instructions."  # type: ignore[index]
    for expected, score in ((INTENT, 1), ("cash_withdrawal_charge", 0)):
        case = Case(id="1", input={"text": TEXT}, expected={"intent": expected})
        scored = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
        assert scored.objective == score
    with pytest.raises(ValueError, match="do not pass"):
        await integration.run_case(case_input={"text": TEXT}, runtime=SimpleNamespace(model="openai:gpt-4o"))
    with pytest.raises(ValueError):
        await integration.score_case(
            case=case, result=CaseResult(output={"intent": "invented"}), case_files_dir=tmp_path
        )
