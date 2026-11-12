"""Offline tests: every network call goes to a mock transport."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from beaker import Case, CaseResult, RetryableCaseError, RolloutRuntime
from beaker.tracing import current_trace
from openai import AsyncOpenAI

from livenewsbench import cli, data, grading, runner, youcom
from livenewsbench.chat import ChatModel, TransientServiceError
from livenewsbench.tools import BLOCKED_DOMAINS, Page, SearchHit, ToolRefused, WebTools, blocked


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Which city hosted the summit?"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(youcom.asyncio, "sleep", instant)


def load_integration() -> Any:
    spec = importlib.util.spec_from_file_location("beaker_integration", ROOT / ".beaker/beaker_integration.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- data ---------------------------------------------------------------------------


def test_grader_prompt_is_upstream_verbatim() -> None:
    content = (ROOT / "src/livenewsbench/grade_simple_qa_prompt.txt").read_bytes()
    assert hashlib.sha256(content).hexdigest() == "fb6d6c1b4b6546f14f462ddbe19fdca47d04f29bef8b356949e2a44fe8e149d5"


def record(question: str, answer: str = "Geneva") -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "date": "2025-11-02",
        "category": "International relations",
        "articles": [{"title": "Source", "article_content": f"The answer is {answer}."}],
    }


def sources(test_questions: list[str], verified: list[str]) -> dict[str, bytes]:
    files = {
        "train": [record("train question")],
        "val": [record("val question")],
        "test": [record(question) for question in test_questions],
        "human_verified_test": [record(question) for question in verified],
    }
    return {f"{split}.jsonl": "".join(json.dumps(r) + "\n" for r in rows).encode() for split, rows in files.items()}


def test_rows_drop_source_articles_and_flag_verified_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {"train": 1, "val": 1, "test": 2, "human_verified_test": 1})
    splits = data.make_rows(sources(["test a", "test b"], ["test b"]))
    assert {split: len(rows) for split, rows in splits.items()} == {"train": 1, "val": 1, "test": 2}
    test_rows = {row.input.question: row for row in splits["test"]}
    assert not test_rows["test a"].metadata.human_verified
    assert test_rows["test b"].metadata.human_verified
    serialized = splits["test"][0].model_dump_json()
    assert "article" not in serialized and "The answer is" not in serialized
    assert splits["train"][0].id == data.row_id("train question")


def test_rows_reject_verified_questions_outside_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {"train": 1, "val": 1, "test": 1, "human_verified_test": 1})
    with pytest.raises(ValueError, match="subset"):
        data.make_rows(sources(["test a"], ["elsewhere"]))


def test_prepare_writes_beaker_splits_and_human_verified_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "EXPECTED_COUNTS", {"train": 1, "val": 1, "test": 2, "human_verified_test": 1})
    fake = sources(["test a", "test b"], ["test b"])
    monkeypatch.setattr(data, "source_file", lambda name, cache_dir: fake[name])
    counts = data.prepare(tmp_path, tmp_path / "cache")
    assert counts == {"train": 1, "val": 1, "test": 2, "human_verified_test": 1}
    assert data.load_rows(tmp_path, "val")[0].input.question == "val question"
    assert [row.input.question for row in data.load_rows(tmp_path, "human_verified_test")] == ["test b"]
    assert (tmp_path / "optimization/train.jsonl").exists() and (tmp_path / "optimization/test.jsonl").exists()


# --- web rules ------------------------------------------------------------------------


class FakeBackend:
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.searches: list[dict[str, Any]] = []
        self.fetched: list[str] = []

    async def search(
        self, query: str, *, count: int, extraction: str, exclude_domains: tuple[str, ...]
    ) -> list[SearchHit]:
        self.searches.append({"query": query, "count": count, "extraction": extraction, "exclude": exclude_domains})
        return [SearchHit(url=url, title=f"title {url}", snippet="snippet") for url in self.urls]

    async def fetch(self, url: str) -> Page:
        self.fetched.append(url)
        return Page(url=url, title="Page", content="Geneva hosted it.")


async def test_search_budget_is_enforced_before_any_request() -> None:
    backend = FakeBackend(["https://news.example/a"])
    web = WebTools(backend=backend, trace=current_trace())
    for index in range(5):
        await web.search(f"query {index}")
    with pytest.raises(ToolRefused, match="budget"):
        await web.search("one too many")
    assert len(backend.searches) == 5
    assert backend.searches[0]["exclude"] == BLOCKED_DOMAINS


async def test_pages_must_come_from_searches_and_blocked_domains_never_reach_the_agent() -> None:
    backend = FakeBackend(["https://news.example/a", "https://web.archive.org/web/1/x", "https://hf.co/datasets/x"])
    web = WebTools(backend=backend, trace=current_trace())
    hits = await web.search("summit city")
    assert [hit.url for hit in hits] == ["https://news.example/a"]
    with pytest.raises(ToolRefused, match="earlier search"):
        await web.open_page("https://news.example/never-returned")
    with pytest.raises(ToolRefused, match="excluded"):
        await web.open_page("https://web.archive.org/web/1/x")
    page = await web.open_page("https://news.example/a")
    assert page.content == "Geneva hosted it."
    assert backend.fetched == ["https://news.example/a"] and web.usage.page_visits == 1
    for _ in range(4):
        await web.open_page("https://news.example/a")
    with pytest.raises(ToolRefused, match="page-visit budget"):
        await web.open_page("https://news.example/a")


def test_blocked_matches_subdomains_only() -> None:
    assert blocked("https://www.github.com/x") and blocked("https://huggingface.co/datasets")
    assert not blocked("https://notgithub.com/x") and not blocked("https://news.example/archive.org")


# --- You.com client --------------------------------------------------------------------


def youcom_web(handler: Callable[[httpx.Request], httpx.Response]) -> youcom.YouComWeb:
    return youcom.YouComWeb("offline-key", transport=httpx.MockTransport(handler))


async def test_search_request_and_result_parsing() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": {
                    "web": [
                        {
                            "url": "https://a.example",
                            "title": "A",
                            "snippets": ["one", "two"],
                            "page_age": "2025-11-02",
                        },
                        {"url": "https://b.example", "title": "B", "contents": {"highlights": ["hl"]}},
                        {"url": "https://c.example", "title": "C", "description": "desc"},
                    ]
                },
                "metadata": {"latency": 0.3},
            },
        )

    async with youcom_web(handler) as web:
        hits = await web.search("q", count=3, extraction="highlights", exclude_domains=("archive.org",))
    assert [hit.snippet for hit in hits] == ["one\ntwo", "hl", "desc"]
    assert hits[0].published == "2025-11-02"
    request = requests[0]
    assert str(request.url) == youcom.SEARCH_URL and request.headers["x-api-key"] == "offline-key"
    assert json.loads(request.content) == {
        "query": "q",
        "count": 3,
        "exclude_domains": ["archive.org"],
        "extraction": {"extraction_mode": "highlights"},
    }


async def test_contents_request_and_empty_result() -> None:
    bodies: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json=[{"url": "https://a.example", "title": "A", "markdown": "# A"}])
        return httpx.Response(200, json=[])

    async with youcom_web(handler) as web:
        page = await web.fetch("https://a.example")
        missing = await web.fetch("https://b.example")
    assert bodies[0] == {"urls": ["https://a.example"], "formats": ["markdown"]}
    assert (page.title, page.content) == ("A", "# A")
    assert missing.content == "The page could not be fetched."


async def test_rate_limits_are_retried_and_persistent_failures_are_transient() -> None:
    statuses = iter([429, 503, 200])

    def flaky(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, headers={"retry-after": "1"}, json={"results": {"web": []}})

    async with youcom_web(flaky) as web:
        assert await web.search("q", count=1, extraction="snippets", exclude_domains=()) == []

    async with youcom_web(lambda request: httpx.Response(503)) as web:
        with pytest.raises(TransientServiceError):
            await web.search("q", count=1, extraction="snippets", exclude_domains=())

    async with youcom_web(lambda request: httpx.Response(401, json={"detail": "bad key"})) as web:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            await web.search("q", count=1, extraction="snippets", exclude_domains=())


# --- model and agent --------------------------------------------------------------------


def completion(
    content: str | None = None, tool_calls: list[dict[str, Any]] | None = None, reasoning: str | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-v4.1-flash",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 400},
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    }


def call(index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def scripted_router(replies: list[dict[str, Any]], requests: list[dict[str, Any]]) -> AsyncOpenAI:
    script = iter(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.you.com/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(script))

    return AsyncOpenAI(
        base_url=youcom.ROUTER_BASE_URL,
        api_key="offline-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def chat_model(client: AsyncOpenAI) -> ChatModel:
    return ChatModel(
        client=client,
        model=youcom.DEFAULT_MODEL,
        provider="youcom",
        server_address=youcom.ROUTER_HOST,
        max_calls=runner.MAX_MODEL_CALLS,
        trace=current_trace(),
        request_options=youcom.REQUEST_OPTIONS,
    )


async def test_seed_agent_searches_reads_and_answers() -> None:
    requests: list[dict[str, Any]] = []
    replies = [
        completion("Search first.", [call(1, "search", {"query": "summit host city 2025"})], "Need the host city."),
        completion(None, [call(2, "open_page", {"url": "https://news.example/a"})]),
        completion("It was Geneva.\nFinal answer: Geneva"),
    ]
    chat = chat_model(scripted_router(replies, requests))
    backend = FakeBackend(["https://news.example/a"])
    web = WebTools(backend=backend, trace=current_trace())
    agent = runner.load_agent(runner.SEED_AGENT_DIR)
    rollout = await runner.run_rollout(QUESTION, agent=agent, chat=chat, web=web)

    assert (rollout.answer, rollout.finish) == ("Geneva", "answered")
    assert (web.usage.searches, web.usage.page_visits, chat.usage.calls) == (1, 1, 3)
    assert (chat.usage.input_tokens, chat.usage.cached_input_tokens, chat.usage.reasoning_tokens) == (3000, 1200, 180)
    first, last = requests[0], requests[-1]
    assert first["model"] == "deepseek-v4.1-flash" and first["max_tokens"] == 16384
    assert [tool["function"]["name"] for tool in first["tools"]] == ["search", "open_page"]
    assert QUESTION in first["messages"][1]["content"]
    tool_results = [message for message in last["messages"] if message["role"] == "tool"]
    assert "Geneva hosted it." in tool_results[1]["content"]
    assert "1 of 5 searches and 1 of 5 page visits" in tool_results[1]["content"]
    assistant_turns = [message for message in last["messages"] if message["role"] == "assistant"]
    assert assistant_turns[0]["reasoning_content"] == "Need the host city."
    assert assistant_turns[1]["content"] is None and "reasoning_content" not in assistant_turns[1]


async def test_agent_answers_within_the_call_cap_after_budget_refusals() -> None:
    requests: list[dict[str, Any]] = []
    loops = [completion(None, [call(i, "search", {"query": f"q{i}"})]) for i in range(runner.MAX_MODEL_CALLS - 1)]
    chat = chat_model(scripted_router([*loops, completion("Final answer: Geneva")], requests))
    web = WebTools(backend=FakeBackend(["https://news.example/a"]), trace=current_trace())
    rollout = await runner.run_rollout(QUESTION, agent=runner.load_agent(runner.SEED_AGENT_DIR), chat=chat, web=web)
    assert rollout.answer == "Geneva"
    assert len(requests) == runner.MAX_MODEL_CALLS and "tools" not in requests[-1]
    assert web.usage.searches == 5 and web.usage.refused == runner.MAX_MODEL_CALLS - 1 - 5


def test_list_price_uses_router_rates() -> None:
    chat = runner.ChatUsage(calls=2, input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=100_000)
    web = runner.WebUsage(searches=2, page_visits=1)
    price = runner.list_price_usd("deepseek-v4.1-flash", chat, web)
    assert price["model"] == pytest.approx(0.5 * 0.30 + 0.5 * 0.007 + 0.1 * 1.20)
    assert price["total"] == pytest.approx(price["model"] + 0.010 + 0.001)  # type: ignore[operator]
    assert runner.list_price_usd(None, chat, web)["total"] is None


# --- grader -------------------------------------------------------------------------------


def grader_client(replies: list[str], requests: list[dict[str, Any]]) -> AsyncOpenAI:
    script = iter(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        reply = completion(next(script))
        return httpx.Response(200, json=reply)

    return AsyncOpenAI(api_key="offline-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_grader_matches_upstream_request_and_letters() -> None:
    requests: list[dict[str, Any]] = []
    client = grader_client(["A", "B", "C"], requests)
    verdicts = [
        await grading.grade(client, model="gpt-4.1", question=QUESTION, gold="Geneva", predicted=answer)
        for answer in ("Geneva", "Paris", "I don't know")
    ]
    assert verdicts == ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
    request = requests[0]
    assert request["model"] == "gpt-4.1" and request["temperature"] == 0.0 and request["max_completion_tokens"] == 1024
    assert request["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert "Gold target: Geneva\nPredicted answer: Geneva" in request["messages"][1]["content"]


async def test_grader_retries_then_raises_and_skips_empty_answers() -> None:
    requests: list[dict[str, Any]] = []
    client = grader_client(["maybe", "B"], requests)
    assert await grading.grade(client, model="gpt-4.1", question=QUESTION, gold="Geneva", predicted="Paris") == (
        "INCORRECT"
    )
    with pytest.raises(grading.GraderError):
        await grading.grade(
            grader_client(["?", "?", "?"], []), model="gpt-4.1", question=QUESTION, gold="Geneva", predicted="x"
        )
    assert await grading.grade(client, model="gpt-4.1", question=QUESTION, gold="Geneva", predicted=" ") == (
        "NOT_ATTEMPTED"
    )
    assert len(requests) == 2


async def test_evaluate_stops_before_rollouts_when_the_grader_key_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    client = AsyncOpenAI(api_key="bad-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(SystemExit, match="HTTP 401"):
        await cli.check_grader(client)


def test_summary_keeps_rollouts_the_grader_failed_on() -> None:
    usage = {"seconds": 10.0, "model": {"calls": 3}, "web": {"searches": 2, "page_visits": 1}}
    outcomes = [
        {"correct": True, "grade": "CORRECT", "usage": usage, "list_price_usd": {"total": 0.02}},
        {"correct": False, "error": "AuthenticationError", "error_stage": "grading", "usage": usage,
         "list_price_usd": {"total": 0.04}},
        {"correct": False, "error": "TransientServiceError", "error_stage": "rollout"},
    ]  # fmt: skip
    args = argparse.Namespace(model=youcom.DEFAULT_MODEL, base_url=youcom.ROUTER_BASE_URL, split="val")
    summary = cli.summarize(outcomes, args)
    assert (summary["errors"], summary["grading_errors"], summary["accuracy"]) == (2, 1, pytest.approx(1 / 3))
    assert summary["list_price_usd_total"] == pytest.approx(0.06)
    assert summary["list_price_usd_per_question"] == pytest.approx(0.03)


# --- Beaker integration -------------------------------------------------------------------


def candidate_dir(tmp_path: Path, answer: str) -> Path:
    """A candidate checkout whose agent differs from the installed seed."""
    agent_dir = tmp_path / "src/livenewsbench/agent"
    shutil.copytree(runner.SEED_AGENT_DIR, agent_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (agent_dir / "agent.py").write_text(
        f"async def answer_question(question, *, chat, web):\n    await web.search(question)\n    return {answer!r}\n"
    )
    return tmp_path


async def test_run_case_uses_the_candidate_agent_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = load_integration()
    monkeypatch.chdir(candidate_dir(tmp_path, "Candidate answer"))
    monkeypatch.setenv("YDC_API_KEY", "offline-key")
    searches: list[dict[str, Any]] = []

    def search(request: httpx.Request) -> httpx.Response:
        searches.append(json.loads(request.content))
        return httpx.Response(200, json={"results": {"web": []}})

    real_web = youcom.YouComWeb
    monkeypatch.setattr(youcom, "YouComWeb", lambda key: real_web(key, transport=httpx.MockTransport(search)))
    runtime = RolloutRuntime(case_files_dir=tmp_path, trace=current_trace())
    result = await integration.run_case(case_input={"question": QUESTION}, runtime=runtime)
    assert result.output == "Candidate answer" and result.output_kind == "value"
    assert searches[0]["query"] == QUESTION and "archive.org" in searches[0]["exclude_domains"]


async def test_run_case_turns_provider_outages_into_retryable_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration = load_integration()
    monkeypatch.chdir(candidate_dir(tmp_path, "unused"))
    monkeypatch.setenv("YDC_API_KEY", "offline-key")
    real_web = youcom.YouComWeb
    outage = httpx.MockTransport(lambda request: httpx.Response(503))
    monkeypatch.setattr(youcom, "YouComWeb", lambda key: real_web(key, transport=outage))
    runtime = RolloutRuntime(case_files_dir=tmp_path, trace=current_trace())
    with pytest.raises(RetryableCaseError):
        await integration.run_case(case_input={"question": QUESTION}, runtime=runtime)


async def test_score_case_is_one_for_correct_and_explains_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    integration = load_integration()
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    monkeypatch.delenv("BEAKER_RUN_ID", raising=False)
    monkeypatch.delenv("BEAKER_SCORER_MODEL", raising=False)
    seen: dict[str, Any] = {}

    async def fake_grade(client: AsyncOpenAI, **kwargs: Any) -> grading.Grade:
        seen.update(kwargs)
        return "CORRECT" if kwargs["predicted"] == "Geneva" else "INCORRECT"

    monkeypatch.setattr(integration, "grade", fake_grade)
    case = Case(id="lnb-1", input={"question": QUESTION}, expected={"answer": "Geneva"})
    right = await integration.score_case(case=case, result=CaseResult(output="Geneva"), case_files_dir=Path())
    wrong = await integration.score_case(case=case, result=CaseResult(output="Paris"), case_files_dir=Path())
    assert (right.objective, wrong.objective) == (1.0, 0.0)
    assert seen["model"] == "gpt-4.1" and seen["gold"] == "Geneva"
    check = wrong.checks[0]
    assert (check.verdict, check.expected, check.predicted, check.message) == ("fail", "Geneva", "Paris", "INCORRECT")


def test_integration_edits_only_the_agent_package() -> None:
    targets = load_integration().integration.targets
    assert [target.paths for target in targets] == [("src/livenewsbench/agent",)]
