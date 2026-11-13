from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from beaker import Case, CaseResult
from beaker.tracing import current_trace
from openai import AsyncOpenAI

from clinc150_router import cli
from clinc150_router.agent import ChatModel, InvalidAnswer, parse_intent, route, system_prompt
from clinc150_router.contracts import LABELS, OUT_OF_SCOPE, Prediction
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


def chat_client(
    received: list[dict[str, Any]],
    *,
    content: str = json.dumps({"intent": INTENT}),
    finish_reason: str = "stop",
    refusal: str | None = None,
) -> AsyncOpenAI:
    def respond(request: httpx.Request) -> httpx.Response:
        received.append(
            {"url": str(request.url), "auth": request.headers["authorization"], **json.loads(request.content)}
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 0,
                "model": json.loads(request.content)["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": content, "refusal": refusal},
                    }
                ],
                "usage": {"prompt_tokens": 900, "completion_tokens": 7, "total_tokens": 907},
            },
        )

    return AsyncOpenAI(
        api_key="offline-test-key",
        base_url="https://offline.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )


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


def test_seed_prompt_lists_all_151_intents() -> None:
    prompt = system_prompt(ROOT / "prompts")
    assert len(LABELS) == len(set(LABELS)) == 151 and LABELS[-1] == OUT_OF_SCOPE
    assert all(f"\n- {label}: " in prompt for label in LABELS)


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
        system_prompt(write_prompts(tmp_path, intents))


def test_empty_descriptions_and_instructions_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        system_prompt(write_prompts(tmp_path / "a", seed_intents() | {INTENT: "  "}))
    with pytest.raises(ValueError, match="instructions"):
        system_prompt(write_prompts(tmp_path / "b", seed_intents(), instructions=" "))


@pytest.mark.parametrize(
    "content",
    [
        '{"intent": "none_of_these"}',
        '{"intent": null}',
        '{"intent": "translate", "extra": 1}',
        '{"intent": "oos", "intent": "translate"}',
        "translate",
        "",
        None,
        "[]",
    ],
)
def test_unknown_answer_is_an_error_not_oos(content: str | None) -> None:
    with pytest.raises(InvalidAnswer):
        parse_intent(content)


@pytest.mark.parametrize(("finish_reason", "refusal"), [("length", None), ("stop", "Refused")])
async def test_incomplete_or_refused_answer_is_invalid(finish_reason: str, refusal: str | None) -> None:
    async with chat_client([], finish_reason=finish_reason, refusal=refusal) as client:
        prediction = await route(
            TEXT, chat=ChatModel(client, "openai", "m"), prompt="Classify the message.", trace=cli.NoTrace()
        )
    assert prediction.intent is None and prediction.error


# Integration


async def test_run_case_uses_pinned_default_without_a_model_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = integration_module()
    write_prompts(tmp_path / "prompts", seed_intents(), instructions="Candidate-specific routing instructions.")
    monkeypatch.chdir(tmp_path)
    received: list[dict[str, Any]] = []
    monkeypatch.setattr(integration, "AsyncOpenAI", lambda **kwargs: chat_client(received))
    result = await integration.run_case(
        case_input={"text": TEXT}, runtime=SimpleNamespace(model=None, trace=current_trace())
    )
    request = received[0]
    assert request["model"] == "gpt-6-luna" and request["reasoning_effort"] == "low"
    assert request["messages"][0]["content"].startswith("Candidate-specific routing instructions.")
    assert request["messages"][1] == {"role": "user", "content": TEXT}
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["intent"]["enum"] == list(LABELS)
    assert result.output["intent"] == INTENT and result.output["input_tokens"] == 900


async def test_run_case_uses_gateway_model_when_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    integration = integration_module()
    write_prompts(tmp_path / "prompts", seed_intents())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEAKER_INFERENCE_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("BEAKER_INFERENCE_API_KEY", "run-scoped-token")
    received: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> AsyncOpenAI:
        created.append(kwargs)
        return chat_client(received)

    monkeypatch.setattr(integration, "AsyncOpenAI", client)
    spans: list[dict[str, Any]] = []

    class RecordingTrace(cli.NoTrace):
        def model_call(self, **kwargs: Any) -> Any:
            spans.append(kwargs)
            return super().model_call(**kwargs)

    runtime = SimpleNamespace(model="anthropic:claude-sonnet-5-5", trace=RecordingTrace())
    await integration.run_case(case_input={"text": TEXT}, runtime=runtime)
    assert (spans[0]["operation"], spans[0]["provider"], spans[0]["model"]) == (
        "chat",
        "anthropic",
        "claude-sonnet-5-5",
    )
    assert created[0]["base_url"] == "https://gateway.test/v1" and created[0]["api_key"] == "run-scoped-token"
    assert received[0]["model"] == "anthropic:claude-sonnet-5-5"
    assert "reasoning_effort" not in received[0]


async def test_invalid_answer_is_observable_and_scores_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    integration = integration_module()
    write_prompts(tmp_path / "prompts", seed_intents())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(integration, "AsyncOpenAI", lambda **kwargs: chat_client([], content='{"intent": "other"}'))
    result = await integration.run_case(
        case_input={"text": TEXT}, runtime=SimpleNamespace(model=None, trace=current_trace())
    )
    assert result.output["intent"] is None and "unknown intent" in result.output["error"]
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
async def test_scoring_fills_scope_specific_field_scores(
    tmp_path: Path, expected: str, predicted: str, scores: dict[str, float]
) -> None:
    integration = integration_module()
    result = CaseResult(output={"intent": predicted, "model": "m"}, output_kind="record")
    case = Case(id="1", input={"text": TEXT}, expected={"intent": expected})
    scored = await integration.score_case(case=case, result=result, case_files_dir=tmp_path)
    weight = 1 if expected == OUT_OF_SCOPE else 1 / 30
    assert scored.objective == pytest.approx(scores["accuracy"] * weight)
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

    def read_prompt(path: Path) -> str:
        prompts.append(path)
        return system_prompt(path)

    async def fake_route(text: str, **kwargs: Any) -> Any:
        answer = next(answers)
        if answer is None:
            raise TimeoutError
        return Prediction(intent=answer, model="m", input_tokens=10, output_tokens=1, cost_usd=0.001)

    monkeypatch.setattr(cli, "route", fake_route)
    monkeypatch.setattr(cli, "system_prompt", read_prompt)
    monkeypatch.setattr(cli, "time", SimpleNamespace(perf_counter=lambda: next(timestamps)))
    output = tmp_path / "result.jsonl"
    args = argparse.Namespace(
        data=tmp_path, split="validation", limit=None, prompts=ROOT / "prompts", output=output, concurrency=1
    )
    async with chat_client([]) as client:
        metrics = await cli.evaluate(args, ChatModel(client, "openai", "m"))
        assert not client.is_closed()
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
