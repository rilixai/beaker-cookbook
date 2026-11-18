from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx2
import pytest
from backports import zstd
from beaker import Case, CaseResult
from typesafe_sdk import AsyncTypeSafeClient

from clinc150_router import cli
from clinc150_router.agent import question
from clinc150_router.contracts import DEFAULT_MODEL, LABELS, OUT_OF_SCOPE, Prediction
from clinc150_router.data import SPLIT_FILES, UPSTREAM, Expected, Message, Row, make_splits, source_file


ROOT = Path(__file__).resolve().parents[1]
TEXT = "what expression would i use to say i love you if i were an italian"  # First upstream train example.
INTENT = "translate"
IN_SCOPE = [label for label in LABELS if label != OUT_OF_SCOPE]


def integration_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("clinc150_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision_client(received: list[dict[str, Any]], *, intent: str = INTENT) -> AsyncTypeSafeClient:
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        body = request.content
        if request.headers.get("content-encoding") == "zstd":
            body = zstd.decompress(body)
        received.append(json.loads(body))
        return httpx2.Response(
            200,
            json={
                "model": "resolved-decision-model",
                "answers": {
                    "intent": {
                        "type": "choice",
                        "choice": intent,
                        "confidence": 1.0,
                        "probabilities": {intent: 1.0},
                    }
                },
                "usage": {"input_tokens": 900, "output_tokens": 7},
            },
        )

    return AsyncTypeSafeClient(api_key="offline-test-key", transport=httpx2.MockTransport(respond))


def fake_source() -> bytes:
    """Synthetic markers test splitting only; never used as evaluation data."""
    upstream: dict[str, list[list[str]]] = {}
    for in_scope_key, oos_key, per_intent, oos_count in UPSTREAM.values():
        upstream[in_scope_key] = [
            [f"{in_scope_key}-{label}-{i}", label] for label in IN_SCOPE for i in range(per_intent)
        ]
        upstream[oos_key] = [[f"{oos_key}-{i}", OUT_OF_SCOPE] for i in range(oos_count)]
    return json.dumps(upstream).encode()


# Prepare


def test_corrupt_cache_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "data_full.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        source_file(tmp_path)


def test_splits_match_upstream_sizes_are_disjoint_and_deterministic() -> None:
    source = fake_source()
    splits = make_splits(source)
    assert splits == make_splits(source)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 15100, "validation": 3100, "test": 5500}
    validation = Counter(row.expected.intent for row in splits["validation"])
    assert validation[OUT_OF_SCOPE] == 100 and validation[INTENT] == 20 and len(validation) == 151
    ids = [{row.id for row in rows} for rows in splits.values()]
    texts = [{row.input.text for row in rows} for rows in splits.values()]
    assert sum(map(len, ids)) == len(set().union(*ids)) == 23700
    assert sum(map(len, texts)) == len(set().union(*texts))
    # Seeded hash order, not upstream order: a prefix mixes intents.
    assert len({row.expected.intent for row in splits["train"][:300]}) > 100
    assert SPLIT_FILES == {
        "train": "optimization/train.jsonl",
        "validation": "optimization/test.jsonl",
        "test": "heldout/test.jsonl",
    }


def test_wrong_upstream_size_fails() -> None:
    upstream = json.loads(fake_source())
    upstream["oos_val"].pop()
    with pytest.raises(ValueError, match="Unexpected upstream validation"):
        make_splits(json.dumps(upstream).encode())


# Prompt and answer validation


def write_prompts(path: Path, intents: dict[str, str], instructions: str = "Pick one intent.") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "instructions.md").write_text(instructions)
    (path / "intents.json").write_text(json.dumps(intents))
    return path


def seed_intents() -> dict[str, str]:
    intents: dict[str, str] = json.loads((ROOT / "prompts/intents.json").read_text())
    return intents


def test_seed_question_lists_all_151_intents() -> None:
    routing_question = question(ROOT / "prompts")
    assert len(LABELS) == len(set(LABELS)) == 151 and LABELS[-1] == OUT_OF_SCOPE
    assert routing_question.criteria == seed_intents()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop(INTENT),  # missing
        lambda d: d.update(invented_intent="a new label"),  # extra
        lambda d: d.update({"Translate": d.pop(INTENT)}),  # wrong
    ],
)
def test_candidate_cannot_change_intent_ids(tmp_path: Path, change: Any) -> None:
    intents = seed_intents()
    change(intents)
    with pytest.raises(ValueError, match="151 intent IDs"):
        question(write_prompts(tmp_path, intents))


def test_empty_descriptions_and_instructions_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        question(write_prompts(tmp_path / "a", seed_intents() | {INTENT: "  "}))
    with pytest.raises(ValueError, match="instructions"):
        question(write_prompts(tmp_path / "b", seed_intents(), instructions=" "))


# Integration


async def test_run_case_sends_and_traces_native_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    integration = integration_module()
    write_prompts(tmp_path / "prompts", seed_intents(), instructions="Candidate-specific routing instructions.")
    monkeypatch.chdir(tmp_path)
    received: list[dict[str, Any]] = []
    monkeypatch.setattr(integration, "AsyncTypeSafeClient", lambda **kwargs: decision_client(received))
    spans: list[dict[str, Any]] = []

    class RecordingTrace(cli.NoTrace):
        def model_call(self, **kwargs: Any) -> Any:
            spans.append(kwargs)
            return super().model_call(**kwargs)

    result = await integration.run_case(
        case_input={"text": TEXT}, runtime=SimpleNamespace(model=None, trace=RecordingTrace())
    )
    assert len(received) == 1
    request = received[0]
    assert request["model"] == DEFAULT_MODEL == "jev-1.14.0"
    assert request["state"] == TEXT
    assert request["questions"]["intent"] == {
        "type": "choice",
        "instructions": "Candidate-specific routing instructions.",
        "criteria": seed_intents(),
    }
    assert (spans[0]["operation"], spans[0]["provider"], spans[0]["model"]) == ("systemone", "typesafe", DEFAULT_MODEL)
    assert result.output["intent"] == INTENT and result.output["input_tokens"] == 900
    assert result.output["model"] == "resolved-decision-model"


async def test_run_case_rejects_chat_model_override() -> None:
    integration = integration_module()
    with pytest.raises(ValueError, match="Select decision models"):
        await integration.run_case(
            case_input={"text": TEXT}, runtime=SimpleNamespace(model="openai:gpt-6-luna", trace=cli.NoTrace())
        )


async def test_invalid_answer_is_observable_and_scores_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    integration = integration_module()
    write_prompts(tmp_path / "prompts", seed_intents())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(integration, "AsyncTypeSafeClient", lambda **kwargs: decision_client([], intent="other"))
    result = await integration.run_case(
        case_input={"text": TEXT}, runtime=SimpleNamespace(model=None, trace=cli.NoTrace())
    )
    assert result.output["intent"] is None and "known intent" in result.output["error"]
    case = Case(id="1", input={"text": TEXT}, expected={"intent": OUT_OF_SCOPE})
    scored = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    assert scored.objective == 0 and scored.checks[0].message == result.output["error"]


@pytest.mark.parametrize(
    ("expected", "predicted", "scores"),
    [
        (INTENT, INTENT, {"accuracy": 1.0, "in_scope_accuracy": 1.0}),
        (INTENT, OUT_OF_SCOPE, {"accuracy": 0.0, "in_scope_accuracy": 0.0}),
        (OUT_OF_SCOPE, OUT_OF_SCOPE, {"accuracy": 1.0, "oos_recall": 1.0}),
        (OUT_OF_SCOPE, INTENT, {"accuracy": 0.0, "oos_recall": 0.0}),
    ],
)
async def test_scoring_uses_accuracy_as_objective_and_reports_scope_metrics(
    tmp_path: Path, expected: str, predicted: str, scores: dict[str, float]
) -> None:
    integration = integration_module()
    result = CaseResult(output={"intent": predicted, "model": "m"}, output_kind="record")
    case = Case(id="1", input={"text": TEXT}, expected={"intent": expected})
    scored = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    assert scored.objective == scores["accuracy"]
    assert dict(scored.field_scores) == scores
    check = scored.checks[0]
    assert (check.name, check.expected, check.predicted) == ("intent", expected, predicted)


@pytest.mark.parametrize(
    "output",
    [
        {"intent": INTENT, "model": "m", "error": "Incomplete answer."},
        {"intent": INTENT, "model": "m", "extra": True},
        {"intent": "unrecognized", "model": "m"},
    ],
)
async def test_scorer_fails_error_and_invalid_records(tmp_path: Path, output: Any) -> None:
    integration = integration_module()
    case = Case(id="1", input={"text": TEXT}, expected={"intent": INTENT})
    scored = await integration.score_case(case=case, result=CaseResult(output=output), case_files_dir=tmp_path)
    assert scored.objective == scored.field_scores["accuracy"] == 0
    assert scored.checks[0].verdict == "fail" and scored.checks[0].message


# CLI


async def test_cli_keeps_errors_in_denominator_and_reports_oos_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = [INTENT, INTENT, OUT_OF_SCOPE, OUT_OF_SCOPE]
    rows = [
        Row(id=str(i), input=Message(text=TEXT), expected=Expected(intent=label)) for i, label in enumerate(labels)
    ]
    source = tmp_path / SPLIT_FILES["validation"]
    source.parent.mkdir()
    source.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    # Predictions: correct, wrong as oos, correct oos, request error.
    answers = iter([INTENT, OUT_OF_SCOPE, OUT_OF_SCOPE, None])
    prompts = []
    timestamps = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03, 3.0, 3.8])

    def read_question(path: Path) -> Any:
        prompts.append(path)
        return question(path)

    async def fake_route(text: str, **kwargs: Any) -> Any:
        answer = next(answers)
        if answer is None:
            raise TimeoutError
        return Prediction(intent=answer, model="m", input_tokens=10, output_tokens=1, cost_usd=0.001)

    monkeypatch.setattr(cli, "route", fake_route)
    monkeypatch.setattr(cli, "question", read_question)
    monkeypatch.setattr(cli, "time", SimpleNamespace(perf_counter=lambda: next(timestamps)))
    output = tmp_path / "result.jsonl"
    args = argparse.Namespace(
        data=tmp_path, split="validation", limit=None, prompts=ROOT / "prompts", output=output, concurrency=1
    )
    async with decision_client([]) as client:
        metrics = await cli.evaluate(args, client)
    assert prompts == [ROOT / "prompts"]
    assert metrics["cases"] == 4 and metrics["errors"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 4)
    assert metrics["in_scope_accuracy"] == pytest.approx(1 / 2)
    assert metrics["oos_recall"] == pytest.approx(1 / 2)
    assert metrics["oos_precision"] == pytest.approx(1 / 2)
    assert metrics["cost_per_1000_messages_usd"] is None
    assert metrics["latency_p95_ms"] == pytest.approx(800)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 4 and records[-1]["error"] == "TimeoutError"
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["split"] == "validation" and summary["oos_recall"] == metrics["oos_recall"]
    assert cli.summarize(records[:-1])["cost_per_1000_messages_usd"] == pytest.approx(1)
