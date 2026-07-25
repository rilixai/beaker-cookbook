"""Tests for the PydanticAI HotpotQA agent and the local evaluation.

Hermetic — zero network. The summarize tool does a direct
``AsyncOpenAI`` call by default; tests inject a scripted
``summarize_llm_call`` instead. The outer agent (which actually needs
PydanticAI's tool-dispatch loop) is driven by a ``FunctionModel`` that
scripts each turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from hotpotqa.agent.agent import HotpotQAPydanticAgent
from hotpotqa.config import HotpotQAConfig, bare_openai_model, to_pydantic_ai_model
from hotpotqa.data.dataset import HotpotQAParagraph, HotpotQARecord
from hotpotqa.evaluation.local_eval import evaluate_agent_on_records, evaluate_record
from hotpotqa.evaluation.report import eval_summary
from hotpotqa.evaluation.scoring import ANSWER_F1_FIELD, ANSWER_FIELD, SUPPORTING_TITLES_RECALL_FIELD


def _record(*, case_id: str = "case-1", answer: str = "Paris") -> HotpotQARecord:
    return HotpotQARecord(
        case_id=case_id,
        question="Which city has the Eiffel Tower?",
        answer=answer,
        question_type="bridge",
        level="easy",
        paragraphs=(
            HotpotQAParagraph(
                title="Eiffel Tower",
                sentences=("The Eiffel Tower is an iron lattice tower in Paris.",),
            ),
            HotpotQAParagraph(
                title="Paris",
                sentences=("Paris is the capital of France.",),
            ),
            HotpotQAParagraph(
                title="Berlin",
                sentences=("Berlin is the capital of Germany.",),
            ),
        ),
        supporting_titles=("Eiffel Tower", "Paris"),
        supporting_sentence_ids={"Eiffel Tower": (0,), "Paris": (0,)},
    )


def _distractor_config() -> HotpotQAConfig:
    """Config whose retriever is the case's own paragraphs (no network)."""
    return HotpotQAConfig(retrieval_mode="distractor", retrieve_k=2, max_iters=10)


def _count_outer_tool_returns(messages: list[object]) -> int:
    """Count tool returns from the outer agent's own tools (excludes sub-call internals)."""
    outer_tool_names = {"retrieve_k", "summarize"}
    n = 0
    for m in messages:
        for p in getattr(m, "parts", None) or []:
            if type(p).__name__ == "ToolReturnPart" and str(getattr(p, "tool_name", "") or "") in outer_tool_names:
                n += 1
    return n


def _scripted_outer_model() -> FunctionModel:
    """Script the outer agent: retrieve → summarize → retrieve → summarize(with context) → final_result."""

    def _fn(messages: list[object], _info: AgentInfo) -> ModelResponse:
        n = _count_outer_tool_returns(messages)
        if n == 0:
            return ModelResponse(
                parts=[
                    TextPart(content="Retrieve evidence about the Eiffel Tower first."),
                    ToolCallPart(tool_name="retrieve_k", args={"query": "Eiffel Tower"}),
                ]
            )
        if n == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="summarize",
                        args={
                            "question": "Which city has the Eiffel Tower?",
                            "passages": ["Eiffel Tower | iron tower in Paris."],
                        },
                    )
                ]
            )
        if n == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="retrieve_k", args={"query": "Paris capital"})])
        if n == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="summarize",
                        args={
                            "question": "Which city has the Eiffel Tower?",
                            "passages": ["Paris | capital of France."],
                            "context": "Eiffel Tower is in Paris.",
                        },
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"answer": "Paris"})])

    return FunctionModel(_fn)


def _make_scripted_summarize() -> tuple[Any, list[tuple[str, str]]]:
    """Build a scripted ``summarize_llm_call`` and a call-log for inspection."""
    call_log: list[tuple[str, str]] = []

    async def _scripted(system_prompt: str, user_prompt: str) -> str:
        call_log.append((system_prompt, user_prompt))
        if "Prior context" in user_prompt:
            return "Paris is the capital and hosts the Eiffel Tower."
        return "Eiffel Tower is in Paris."

    return _scripted, call_log


def _build_agent(**kwargs: Any) -> tuple[HotpotQAPydanticAgent, list[tuple[str, str]]]:
    scripted, call_log = _make_scripted_summarize()
    kwargs.setdefault("summarize_llm_call", scripted)
    agent = HotpotQAPydanticAgent(
        model=_scripted_outer_model(),
        top_k=2,
        max_iters=10,
        **kwargs,
    )
    return agent, call_log


# ─── agent ──────────────────────────────────────────────────────────────


def test_agent_runs_end_to_end_with_two_tools_and_structured_output() -> None:
    agent, _calls = _build_agent()

    output = asyncio.run(agent.forward(record=_record()))
    assert output.answer == "Paris"
    tool_names = [tc.tool_name for tc in output.tool_calls]
    assert tool_names.count("retrieve_k") == 2
    assert tool_names.count("summarize") == 2
    assert tool_names[-1] == "finish"
    # At least one summarize call carried ``context`` — the single-tool
    # signature lets the agent decide when to pass prior context.
    summarize_calls = [tc for tc in output.tool_calls if tc.tool_name == "summarize"]
    contexted = [tc for tc in summarize_calls if "context" in tc.tool_args]
    assert contexted, "expected at least one summarize call with a context arg"


def test_summarize_tool_uses_the_configured_summarize_prompt() -> None:
    captured_prompts: list[str] = []

    async def _capture(system_prompt: str, _user_prompt: str) -> str:
        captured_prompts.append(system_prompt)
        return "ok"

    agent, _ = _build_agent(summarize_llm_call=_capture, summarize_prompt="MY-SUMMARIZE-PROMPT")
    asyncio.run(agent.forward(record=_record()))
    assert captured_prompts and set(captured_prompts) == {"MY-SUMMARIZE-PROMPT"}


def test_policy_prompt_is_visible_to_the_llm_as_the_system_prompt() -> None:
    """The configured ``policy_prompt`` must *observably reach the LLM*.

    PydanticAI silently drops ``Agent.iter(instructions=...)`` in our pinned
    version; only the constructor-bound ``system_prompt`` reaches the model.
    Drive a ``FunctionModel`` that captures every ``SystemPromptPart`` it
    receives and assert the configured prompt shows up there.
    """
    captured_system: list[str] = []

    def model_fn(messages: list[object], _info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in getattr(msg, "parts", None) or []:
                if type(part).__name__ == "SystemPromptPart":
                    captured_system.append(str(getattr(part, "content", "") or ""))
        # Terminate immediately so the test doesn't need a tool-call loop.
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"answer": "X"})])

    agent = HotpotQAPydanticAgent(
        model=FunctionModel(model_fn),
        top_k=1,
        max_iters=2,
        summarize_llm_call=lambda _s, _u: asyncio.sleep(0, result=""),
        policy_prompt="MY-CUSTOM-POLICY-PROMPT",
    )
    asyncio.run(agent.forward(record=_record()))
    assert captured_system, "FunctionModel never saw a SystemPromptPart"
    assert "MY-CUSTOM-POLICY-PROMPT" in captured_system[-1]


def test_agent_defers_inner_build_until_first_use() -> None:
    """Constructing the agent must not build the pydantic-ai Agent eagerly.

    Building it from a model *string* instantiates the provider client (e.g.
    ``AsyncOpenAI()``), which raises without ``OPENAI_API_KEY``. Deferring the
    build keeps every offline code path network- and key-free.
    """
    agent = HotpotQAPydanticAgent(
        model="openai:gpt-4.1-mini",
        top_k=2,
        max_iters=4,
    )
    assert agent._agent is None  # noqa: SLF001 — asserting the lazy-build invariant


def test_agent_retrieval_prefers_the_injected_retriever() -> None:
    """``_do_retrieve`` must honor an injected ``retrieve_k_fn``.

    Regression: it used to always run ``bm25_top_k`` over the case's local
    10 paragraphs (``deps.paragraphs``), ignoring
    ``HotpotQAConfig.retrieval_mode``. So a nominal ``--retrieval fullwiki``
    run silently searched only the local distractor context instead of the
    full Wikipedia dump, producing non-comparable numbers.
    """
    from hotpotqa.agent.agent import HotpotQADeps

    sentinel_paragraph = HotpotQAParagraph(
        title="GlobalCorpusHit",
        sentences=("This paragraph exists only in the injected retriever, not in deps.paragraphs.",),
    )

    calls: list[tuple[str, int]] = []

    def _injected_retriever(query: str, k: int) -> list[HotpotQAParagraph]:
        calls.append((query, k))
        return [sentinel_paragraph]

    agent, _ = _build_agent()
    record = _record()
    deps = HotpotQADeps(
        paragraphs=list(record.paragraphs),  # local distractor paragraphs
        retrieve_k=2,
        gold_supporting_titles=list(record.supporting_titles),
        retrieve_k_fn=_injected_retriever,
    )
    obs = agent._do_retrieve(deps, "Eiffel Tower")

    # Retrieved set came from the injected fn, not from deps.paragraphs.
    assert calls == [("Eiffel Tower", 2)]
    assert [p.title for p in deps.retrieved] == ["GlobalCorpusHit"]
    assert "GlobalCorpusHit" in obs

    # With no injected fn, the agent falls back to bm25 over deps.paragraphs.
    # None of the case's paragraph titles equal "GlobalCorpusHit", so the
    # local path cannot accidentally produce it.
    deps_legacy = HotpotQADeps(
        paragraphs=list(record.paragraphs),
        retrieve_k=2,
        gold_supporting_titles=list(record.supporting_titles),
    )
    obs_legacy = agent._do_retrieve(deps_legacy, "Eiffel Tower")
    assert "GlobalCorpusHit" not in obs_legacy
    assert all(p.title in {"Eiffel Tower", "Paris", "Berlin"} for p in deps_legacy.retrieved)


def test_model_name_normalization_is_centralized_in_config() -> None:
    """Slash- and colon-form model specs both canonicalize to the PydanticAI
    colon form on the config (the single normalization layer), and the bare
    OpenAI name is derived from either separator."""
    assert to_pydantic_ai_model("openai/gpt-4.1-mini") == "openai:gpt-4.1-mini"
    assert to_pydantic_ai_model("openai:gpt-4.1-mini") == "openai:gpt-4.1-mini"
    assert bare_openai_model("openai:gpt-4.1-mini") == "gpt-4.1-mini"
    assert bare_openai_model("openai/gpt-4.1-mini") == "gpt-4.1-mini"
    assert bare_openai_model("gpt-4.1-mini") == "gpt-4.1-mini"

    # A slash-form spec reaching the config is rewritten to the valid
    # PydanticAI colon form rather than passed through.
    cfg = HotpotQAConfig(
        retrieval_mode="distractor", retrieve_k=1, max_iters=2, pydantic_agent_model="openai/gpt-4.1-mini"
    )
    assert cfg.pydantic_agent_model == "openai:gpt-4.1-mini"


# ─── local evaluation ───────────────────────────────────────────────────


def test_evaluate_record_scores_a_correct_answer() -> None:
    agent, _ = _build_agent()
    result = asyncio.run(evaluate_record(record=_record(), agent=agent, config=_distractor_config()))
    assert result["kind"] == "scored"
    assert result["field_scores"][ANSWER_FIELD] == pytest.approx(1.0)
    assert result["field_scores"][ANSWER_F1_FIELD] == pytest.approx(1.0)
    assert result["field_scores"][SUPPORTING_TITLES_RECALL_FIELD] == pytest.approx(1.0)
    assert result["objective"] == pytest.approx(1.0)


def test_evaluate_record_scores_a_wrong_answer_zero() -> None:
    agent, _ = _build_agent()
    result = asyncio.run(evaluate_record(record=_record(answer="Berlin"), agent=agent, config=_distractor_config()))
    assert result["field_scores"][ANSWER_FIELD] == pytest.approx(0.0)
    assert result["objective"] == pytest.approx(0.0)


def test_evaluate_agent_on_records_aggregates() -> None:
    agent, _ = _build_agent()
    records = [_record(case_id=f"case-{i}") for i in range(3)]
    report = asyncio.run(
        evaluate_agent_on_records(agent=agent, records=records, config=_distractor_config(), max_concurrency=2)
    )
    assert report.num_cases == 3
    assert report.num_scored == 3
    assert report.num_errored == 0
    assert report.objective == pytest.approx(1.0)
    assert report.field_accuracies[ANSWER_FIELD] == pytest.approx(1.0)
    summary = eval_summary(report, split="test")
    assert summary["split"] == "test"
    assert summary["num_cases"] == 3
    assert summary["field_sample_counts"][ANSWER_FIELD] == 3


def test_evaluate_agent_bounds_concurrency() -> None:
    """``max_concurrency`` must cap how many cases are in flight at once."""
    in_flight = 0
    peak = 0

    class _SlowAgent(HotpotQAPydanticAgent):
        async def forward(self, *, record: HotpotQARecord, retrieve_k_fn: Any = None) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.01)
                return await super().forward(record=record, retrieve_k_fn=retrieve_k_fn)
            finally:
                in_flight -= 1

    scripted, _ = _make_scripted_summarize()
    agent = _SlowAgent(model=_scripted_outer_model(), top_k=2, max_iters=10, summarize_llm_call=scripted)
    records = [_record(case_id=f"case-{i}") for i in range(6)]
    report = asyncio.run(
        evaluate_agent_on_records(agent=agent, records=records, config=_distractor_config(), max_concurrency=2)
    )
    assert report.num_cases == 6
    assert peak <= 2, f"expected at most 2 concurrent cases, saw {peak}"
    assert peak > 1, "expected the batch to actually run cases concurrently"


def test_evaluate_agent_contains_errors_and_excludes_unscoreable() -> None:
    """One case erroring counts as 0 (deflates) and never aborts the batch; an
    unscoreable case (no gold answer, no gold titles) is excluded from the
    denominator rather than counted as a failure."""
    unscoreable = HotpotQARecord(
        case_id="case-unscoreable",
        question="Which city has the Eiffel Tower?",
        answer="",
        question_type="bridge",
        level="easy",
        paragraphs=(HotpotQAParagraph(title="Paris", sentences=("Paris is the capital of France.",)),),
        supporting_titles=(),
    )
    records = [_record(case_id="case-ok"), _record(case_id="case-boom"), unscoreable]

    class _FlakyAgent(HotpotQAPydanticAgent):
        async def forward(self, *, record: HotpotQARecord, retrieve_k_fn: Any = None) -> Any:
            if record.case_id == "case-boom":
                raise RuntimeError("boom")  # forces that case to error
            return await super().forward(record=record, retrieve_k_fn=retrieve_k_fn)

    scripted, _ = _make_scripted_summarize()
    flaky = _FlakyAgent(
        model=_scripted_outer_model(),
        top_k=2,
        max_iters=10,
        summarize_llm_call=scripted,
    )
    report = asyncio.run(
        evaluate_agent_on_records(agent=flaky, records=records, config=_distractor_config(), max_concurrency=2)
    )
    assert report.num_cases == 3
    assert report.num_errored == 1
    assert report.num_unscoreable == 1
    assert report.num_scored == 1
    # denominator = scored (1) + errored (1) = 2; the errored case scores 0.
    assert report.objective == pytest.approx(0.5)
    assert report.field_accuracies[ANSWER_FIELD] == pytest.approx(0.5)
    errored = next(r for r in report.per_case if r["case_id"] == "case-boom")
    assert errored["kind"] == "error"
    assert "RuntimeError: boom" in errored["error"]
