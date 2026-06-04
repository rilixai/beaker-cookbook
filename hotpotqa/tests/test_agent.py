"""Tests for the idiomatic PydanticAI HotpotQA agent.

The summarize tool does a direct ``AsyncOpenAI`` call by default; tests
inject a scripted ``summarize_llm_call`` instead so no marker-dispatch
trick on the outer model is needed. The outer agent (which actually
needs PydanticAI's tool-dispatch loop) is driven by a ``FunctionModel``
that scripts each turn.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from hotpotqa.agent.agent import HotpotQAPydanticAgent
from hotpotqa.data.dataset import HotpotQAParagraph, HotpotQARecord


def _record() -> HotpotQARecord:
    return HotpotQARecord(
        case_id="case-1",
        question="Which city has the Eiffel Tower?",
        answer="Paris",
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


def _make_scripted_summarize() -> tuple[object, list[tuple[str, str]]]:
    """Build a scripted ``summarize_llm_call`` and a call-log for inspection."""
    call_log: list[tuple[str, str]] = []

    async def _scripted(system_prompt: str, user_prompt: str) -> str:
        call_log.append((system_prompt, user_prompt))
        if "Prior context" in user_prompt:
            return "Paris is the capital and hosts the Eiffel Tower."
        return "Eiffel Tower is in Paris."

    return _scripted, call_log


def _build_agent(
    *,
    policy_prompt: str | None = None,
    summarize_prompt: str | None = None,
    summarize_llm_call: object | None = None,
) -> tuple[HotpotQAPydanticAgent, list[tuple[str, str]]]:
    scripted, call_log = _make_scripted_summarize()
    if summarize_llm_call is None:
        summarize_llm_call = scripted
    agent = HotpotQAPydanticAgent(
        model=_scripted_outer_model(),
        top_k=2,
        max_iters=10,
        policy_prompt=policy_prompt or "You are a careful QA agent.",
        summarize_prompt=summarize_prompt or "Summarize the passages for the question.",
        summarize_llm_call=summarize_llm_call,
    )
    return agent, call_log


def test_agent_runs_end_to_end_with_two_tools_and_structured_output() -> None:
    record = _record()
    agent, _calls = _build_agent()

    output = asyncio.run(
        agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
        )
    )
    assert output.answer == "Paris"
    tool_names = [tc.tool_name for tc in output.tool_calls]
    assert tool_names.count("retrieve_k") == 2
    assert tool_names.count("summarize") == 2
    assert tool_names[-1] == "finish"
    # At least one summarize call carried ``context`` — the simplified
    # single-tool signature lets the agent decide when to pass prior
    # context.
    summarize_calls = [tc for tc in output.tool_calls if tc.tool_name == "summarize"]
    contexted = [tc for tc in summarize_calls if "context" in tc.tool_args]
    assert contexted, "expected at least one summarize call with a context arg"


def test_do_summarize_uses_constructor_prompt() -> None:
    """The summarize tool uses the prompt supplied when the agent is built."""
    from hotpotqa.agent.agent import HotpotQADeps

    captured_prompts: list[str] = []

    async def _capture(system_prompt: str, _user_prompt: str) -> str:
        captured_prompts.append(system_prompt)
        return "ok"

    agent, _ = _build_agent(summarize_prompt="INSTANCE-STATE PROMPT", summarize_llm_call=_capture)

    deps = HotpotQADeps(
        paragraphs=[],
        retrieve_k=2,
        gold_supporting_titles=[],
        # No override — should fall back.
    )
    asyncio.run(agent._do_summarize(deps, question="q?", passages=["p"], context=None))
    assert captured_prompts == ["INSTANCE-STATE PROMPT"]


def test_constructor_policy_prompt_reaches_llm_end_to_end() -> None:
    """The rewritten ``policy_prompt`` must observably reach the LLM.

    Regression context: PydanticAI silently drops ``Agent.iter(instructions=...)``
    in our pinned version; only the constructor-bound ``system_prompt``
    reaches the model. The runner therefore constructs an agent with the
    candidate policy prompt instead of trying to mutate an existing agent.

    This test drives a ``FunctionModel`` that captures every
    ``SystemPromptPart`` it receives. When the agent is constructed with the
    rewrite, that text must appear in the captured stream. We deliberately use a
    ``FunctionModel`` (which sees the same messages the LLM would) so
    the test is robust to PydanticAI version changes that might rename
    internal attributes like ``_system_prompts``.
    """
    captured_system: list[str] = []

    def model_fn(messages: list[object], _info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in getattr(msg, "parts", None) or []:
                if type(part).__name__ == "SystemPromptPart":
                    captured_system.append(str(getattr(part, "content", "") or ""))
        # Terminate immediately so the test doesn't need a tool-call loop.
        return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"answer": "X"})])

    record = _record()
    seed_agent = HotpotQAPydanticAgent(
        model=FunctionModel(model_fn),
        top_k=1,
        max_iters=2,
        summarize_llm_call=lambda _s, _u: asyncio.sleep(0, result=""),
    )

    asyncio.run(
        seed_agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
        )
    )
    assert captured_system, "FunctionModel never saw a SystemPromptPart"
    seed_system = captured_system[-1]
    captured_system.clear()

    rewritten_agent = HotpotQAPydanticAgent(
        model=FunctionModel(model_fn),
        top_k=1,
        max_iters=2,
        policy_prompt="BRAND_NEW_POLICY_FROM_GEPA_REWRITE",
        summarize_llm_call=lambda _s, _u: asyncio.sleep(0, result=""),
    )
    asyncio.run(
        rewritten_agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
        )
    )
    assert captured_system, "no system prompts captured from rewritten agent"
    new_system = captured_system[-1]
    assert "BRAND_NEW_POLICY_FROM_GEPA_REWRITE" in new_system, (
        f"rewritten policy_prompt was NOT visible to the LLM. "
        f"Captured system prompt: {new_system!r}; seed was {seed_system!r}"
    )
    assert "BRAND_NEW_POLICY_FROM_GEPA_REWRITE" not in seed_system, (
        f"Rewrite leaked into the seed-time system prompt: {seed_system!r}"
    )


def test_summarize_feedback_reads_pydantic_ai_tool_arg_names() -> None:
    """``_format_summarize_call`` must read tool args by their parameter names.

    Regression: an earlier version read ``has_context`` and
    ``num_passages`` — keys present in the agent's internal
    ``deps.tool_invocations`` log, NOT in ``step.tool_args``.
    PydanticAI serializes tool args using the actual parameter names,
    so the agent's ``summarize(question, passages, context=None)``
    signature lands on ``step.tool_args`` as ``{"question": ...,
    "passages": [...], ["context": ...]}``. The mismatch made every
    summarize call render as ``num_passages=?, has_context=False``,
    silently misleading the reflection LM.
    """
    from hotpotqa.agent.types import AgentToolCall
    from hotpotqa.feedback import _format_summarize_call

    step_with_context = AgentToolCall(
        step_index=0,
        tool_name="summarize",
        tool_args={
            "question": "Who?",
            "passages": ["P1", "P2", "P3"],
            "context": "earlier summary",
        },
        observation="some summary text",
        thought="",
    )
    rendered = _format_summarize_call(1, step_with_context)
    assert "num_passages=3" in rendered, f"expected 'num_passages=3', got: {rendered}"
    assert "has_context=True" in rendered, f"expected 'has_context=True', got: {rendered}"

    step_no_context = AgentToolCall(
        step_index=1,
        tool_name="summarize",
        tool_args={"question": "What?", "passages": ["only one"]},
        observation="",
        thought="",
    )
    rendered = _format_summarize_call(2, step_no_context)
    assert "num_passages=1" in rendered
    assert "has_context=False" in rendered


def test_agent_feedback_correctness_matches_official_em_scorer() -> None:
    """Regression: agent feedback used a local `_normalize` that only
    lowercased + collapsed whitespace. The official HotpotQA scorer
    (``normalize_answer`` in ``hotpot_eval.py``) additionally strips
    articles (``a/an/the``) and ASCII punctuation. The mismatch let the
    "correct"/"incorrect" label in agent feedback disagree with the
    actual EM metric — e.g. ``"a dog"`` vs gold ``"dog"`` would label
    "incorrect" in feedback but score correct under EM. The reflection
    LM would then try to "fix" prompts that were already producing
    metric-correct answers.

    Switched to ``exact_match_score`` (which uses ``normalize_answer``)
    so feedback labels agree with the scored metric for every case.
    """
    from hotpotqa.agent.types import AgentToolCall, HotpotQAAgentOutput
    from hotpotqa.feedback import (
        _policy_prompt_feedback,
        _summarize_feedback,
    )

    record = HotpotQARecord(
        case_id="case-em",
        question="What animal is a dog?",
        answer="dog",
        question_type="bridge",
        level="easy",
        paragraphs=(HotpotQAParagraph(title="Dog", sentences=("A dog is a mammal.",)),),
        supporting_titles=("Dog",),
        supporting_sentence_ids={"Dog": (0,)},
    )
    # Agent surfaces "a dog" — equal to "dog" under official EM
    # normalization (article stripped) but NOT under the prior local
    # ``_normalize`` (lowercase + whitespace only). Include a summarize
    # call so the summarize feedback hits the branch that interpolates
    # the correctness label (the no-call branch doesn't render it).
    output = HotpotQAAgentOutput(
        answer="a dog",
        retrieved_paragraphs=[],
        tool_calls=[
            AgentToolCall(
                step_index=0,
                tool_name="summarize",
                tool_args={"question": "What animal is a dog?", "passages": ["A dog is a mammal."]},
                observation="A dog is a mammal.",
                thought="",
            )
        ],
    )

    summarize_fb = _summarize_feedback(record=record, output=output)
    policy_fb = _policy_prompt_feedback(record=record, output=output)

    # Both feedback strings interpolate `correctness`. Under the fix they
    # must say "(correct)" — matching what EM scores — never "(incorrect)".
    assert "(correct)" in summarize_fb
    assert "(incorrect)" not in summarize_fb
    assert "(correct)" in policy_fb
    assert "(incorrect)" not in policy_fb


def test_agent_runtime_dispatches_retrieval_by_cfg_mode() -> None:
    """Regression: the agent's ``_do_retrieve`` used to always run
    ``bm25_top_k`` over the case's local 10 paragraphs (``deps.paragraphs``),
    ignoring ``HotpotQAPipelineConfig.retrieval_mode``. So a nominal
    ``--mode pydantic_agent --retrieval fullwiki`` run silently searched
    only the local distractor context instead of the full Wikipedia
    dump, producing non-comparable numbers vs the workflow's fullwiki
    baseline.

    The runtime now builds a per-case retrieve fn via
    ``build_retrieve_k_fn_for_case`` (same path the workflow uses) and
    threads it into ``HotpotQADeps.retrieve_k_fn``; the agent's
    ``_do_retrieve`` honors it. Verify by injecting a sentinel
    ``retrieve_k_fn`` and asserting the agent's retrieved paragraphs
    come from it, not from ``deps.paragraphs``.
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
    # Use the actual ``_do_retrieve`` directly with a hand-built
    # ``HotpotQADeps`` carrying the injected fn — bypasses the
    # PydanticAI run loop so we can deterministically inspect the
    # routing decision.
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

    # Sanity-check the legacy path: with no injected fn, the agent
    # falls back to bm25 over deps.paragraphs. None of the case's
    # paragraph titles equal "GlobalCorpusHit", so the local path
    # cannot accidentally produce it.
    deps_legacy = HotpotQADeps(
        paragraphs=list(record.paragraphs),
        retrieve_k=2,
        gold_supporting_titles=list(record.supporting_titles),
    )
    obs_legacy = agent._do_retrieve(deps_legacy, "Eiffel Tower")
    assert "GlobalCorpusHit" not in obs_legacy
    assert all(p.title in {"Eiffel Tower", "Paris", "Berlin"} for p in deps_legacy.retrieved)
