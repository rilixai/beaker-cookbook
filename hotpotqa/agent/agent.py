"""PydanticAI HotpotQA agent — an idiomatic tool-using agent.

Shipped as the kind of agent a customer would actually write in
PydanticAI to answer multi-hop Wikipedia questions: two tools, a
structured output type, and two prompt strings callers may override.

* **Tools**
  * ``retrieve_k(query)`` — deterministic BM25 / fullwiki paragraph
    retrieval.
  * ``summarize(question, passages, context=None)`` — LLM-backed
    summarization. The agent decides whether to call it (and whether
    to pass a previous summary as ``context``).

* **Prompt knobs**
  * ``policy_prompt`` → the agent's ``system_prompt`` (tool-use policy).
  * ``summarize_prompt`` → injected as the ``system`` message in the
    summarize tool's direct ``chat.completions`` call.

* **Terminator** — a Pydantic output type
  (:class:`HotpotQAOutput.answer`). PydanticAI's built-in
  ``final_result`` mechanism ends the loop once the model emits a valid
  output; there's no ``finish`` tool.

The summarize tool deliberately uses a raw ``AsyncOpenAI`` call rather
than a sub-``pydantic_ai.Agent`` so the optimization story stays
transparent: the ``system_prompt`` GEPA rewrites is right there in the
``messages=[...]`` list. Callers wanting a different provider inject a
``summarize_llm_call`` closure; tests inject a scripted stand-in.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ..data.dataset import HotpotQAParagraph
from ..data.gold import remaining_gold_titles
from .prompts import (
    DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
    DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
)
from .retrieval import RetrieveKFn, bm25_top_k
from .types import AgentToolCall, HotpotQAAgentOutput


logger = logging.getLogger(__name__)


SummarizeLLMCall = Callable[[str, str], Awaitable[str]]
"""Async callable: ``(system_prompt, user_prompt) -> summary_text``.

Pass one to :class:`HotpotQAPydanticAgent` to override the default
``AsyncOpenAI`` call. Tests inject scripted stand-ins.
"""


class HotpotQAOutput(BaseModel):
    """Pydantic output type; the model populates ``answer`` to terminate the loop."""

    answer: str = Field(
        description="Final short-form answer to the multi-hop question (noun phrase, entity, or yes/no).",
    )


@dataclass
class HotpotQADeps:
    """Per-case state the tool functions read and write through ``RunContext``."""

    paragraphs: list[HotpotQAParagraph]
    retrieve_k: int
    gold_supporting_titles: list[str]
    retrieved: list[HotpotQAParagraph] = field(default_factory=list)
    retrieved_titles_seen: set[str] = field(default_factory=set)
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    # Optional retrieval indirection. When set, ``_do_retrieve`` calls
    # this function instead of running bm25 over ``paragraphs`` —
    # letting the runtime dispatch by ``cfg.retrieval_mode`` (fullwiki
    # bm25s index vs case-local distractor BM25) without the agent
    # itself caring about modes. The agent still applies its own
    # cross-call dedup against ``retrieved_titles_seen``.
    retrieve_k_fn: RetrieveKFn | None = None
    # Snapshotted summarize prompt for this case. ``forward`` captures
    # ``self._current_summarize_prompt`` here at the start of the run
    # so a concurrent prompt update can't swap the prompt mid-call
    # (matching the existing ``self._agent`` snapshot in ``forward``).
    # ``_do_summarize`` prefers this when set; direct test callers that build
    # their own ``HotpotQADeps`` can leave it ``None`` and fall back to instance
    # state.
    summarize_prompt_override: str | None = None


class HotpotQAPydanticAgent:
    """Idiomatic PydanticAI HotpotQA agent.

    The agent decides how many times to call ``retrieve_k`` and
    ``summarize`` (with or without ``context``) before populating
    :class:`HotpotQAOutput` to terminate. The summarize tool does a
    direct ``AsyncOpenAI.chat.completions.create`` call by default so
    the system-prompt injection is plainly visible; callers can inject
    a custom :data:`SummarizeLLMCall` to swap providers or hook in a
    test double.
    """

    def __init__(
        self,
        *,
        model: str | Model,
        top_k: int = 7,
        max_iters: int = 8,
        temperature: float = 0.0,
        policy_prompt: str = DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
        summarize_prompt: str = DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
        summarize_model: str = "gpt-4.1-mini",
        summarize_llm_call: SummarizeLLMCall | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        # ``top_k`` rather than ``retrieve_k`` to avoid shadowing the
        # ``retrieve_k`` tool closure registered below.
        self.top_k = top_k
        self.max_iters = max_iters
        # Pinned at construction time and applied identically to the outer
        # Agent's model settings and the raw ``AsyncOpenAI`` summarize call,
        # so the two LLM call sites can't drift apart.
        self.temperature = temperature

        self._summarize_model = summarize_model
        # Defer ``AsyncOpenAI()`` construction until the default summarize
        # call actually runs — its constructor raises if ``OPENAI_API_KEY``
        # is missing, and tests that inject a ``summarize_llm_call``
        # stand-in should never trip that.
        self._openai_client: AsyncOpenAI | None = openai_client
        self._summarize_llm_call: SummarizeLLMCall = summarize_llm_call or self._default_summarize_llm_call

        self._current_policy_prompt = policy_prompt
        self._current_summarize_prompt = summarize_prompt
        # ``set_prompts`` rebuilds the inner Agent when the policy prompt
        # changes (see _build_agent for why); concurrent callers can safely
        # update/read prompt state through this lock.
        self._build_lock = threading.Lock()

        self._agent = self._build_agent()

    def _build_agent(self) -> Agent[HotpotQADeps, HotpotQAOutput]:
        """Construct a fresh inner ``pydantic_ai.Agent`` with the current policy prompt.

        PydanticAI bakes ``system_prompt`` into the agent at construction
        time. We verified empirically that ``Agent.iter(instructions=...)``
        does NOT replace the constructor-bound ``system_prompt`` — the
        ``instructions`` argument is silently dropped, only the original
        ``system_prompt`` reaches the model. So when ``set_prompts`` updates
        the policy prompt, the only correct way to surface that
        change is to rebuild the Agent.

        Tool registration is cheap (a few Python objects) — done on every
        rebuild but never on every ``forward`` call, since rebuilds
        happen at most once per prompt update.
        """
        agent: Agent[HotpotQADeps, HotpotQAOutput] = Agent(
            self._model,
            output_type=HotpotQAOutput,
            deps_type=HotpotQADeps,
            system_prompt=self._current_policy_prompt,
            model_settings=ModelSettings(temperature=self.temperature),
        )

        @agent.tool
        async def retrieve_k(ctx: RunContext[HotpotQADeps], query: str) -> str:
            """Retrieve the top-k Wikipedia paragraphs matching the query."""
            return self._do_retrieve(ctx.deps, query)

        @agent.tool
        async def summarize(
            ctx: RunContext[HotpotQADeps],
            question: str,
            passages: list[str],
            context: str | None = None,
        ) -> str:
            """Summarize the given passages relative to the question.

            Pass ``context`` (a previous summary or relevant prior text)
            when the summary should build on prior reasoning; omit it
            for a fresh summary of newly retrieved passages.
            """
            return await self._do_summarize(ctx.deps, question=question, passages=passages, context=context)

        return agent

    def set_prompts(
        self,
        *,
        policy_prompt: str | None = None,
        summarize_prompt: str | None = None,
    ) -> None:
        """Update prompt strings used by subsequent cases.

        The policy prompt is baked into the inner ``pydantic_ai.Agent``
        at construction time (PydanticAI's ``Agent.iter(instructions=)``
        is dropped silently in our pinned version), so changing
        ``policy_prompt`` triggers an Agent rebuild. The summarize prompt is
        read at tool-call time from instance state, so no rebuild is needed.

        The build lock makes repeated calls with the same policy prompt a
        no-op. A concurrent ``forward`` snapshots ``self._agent`` at the start
        of the call, so swapping the reference mid-batch can't corrupt an
        in-flight run.
        """
        with self._build_lock:
            if policy_prompt is not None and policy_prompt != self._current_policy_prompt:
                self._current_policy_prompt = policy_prompt
                self._agent = self._build_agent()
            if summarize_prompt is not None:
                self._current_summarize_prompt = summarize_prompt

    @property
    def current_policy_prompt(self) -> str:
        return self._current_policy_prompt

    @property
    def current_summarize_prompt(self) -> str:
        return self._current_summarize_prompt

    async def forward(
        self,
        *,
        question: str,
        paragraphs: Sequence[HotpotQAParagraph],
        gold_supporting_titles: Sequence[str] | None = None,
        retrieve_k_fn: RetrieveKFn | None = None,
    ) -> HotpotQAAgentOutput:
        """Run one case through the agent.

        ``retrieve_k_fn`` is the mode-dispatched retriever the runtime
        builds from ``cfg.retrieval_mode``. When omitted (e.g. direct
        test callers), ``_do_retrieve`` falls back to running bm25 over
        ``paragraphs`` — the legacy local-context path.
        """
        from pydantic_ai.usage import UsageLimits

        # Snapshot the Agent reference AND the summarize prompt: if
        # ``set_prompts`` runs on another thread mid-call and rebuilds
        # ``self._agent`` or rewrites
        # ``self._current_summarize_prompt``, this in-flight run keeps
        # a consistent agent + consistent summarize prompt until
        # completion. The summarize snapshot rides through ``deps`` so
        # ``_do_summarize`` (invoked indirectly via PydanticAI's tool
        # dispatch) reads the same value the case started with.
        agent = self._agent
        summarize_prompt_snapshot = self._current_summarize_prompt

        deps = HotpotQADeps(
            paragraphs=list(paragraphs),
            retrieve_k=self.top_k,
            gold_supporting_titles=list(gold_supporting_titles or []),
            retrieve_k_fn=retrieve_k_fn,
            summarize_prompt_override=summarize_prompt_snapshot,
        )

        answer = ""
        messages: list[Any] = []
        try:
            # NOTE: do NOT pass ``instructions=`` here. PydanticAI's
            # ``Agent.iter(instructions=...)`` is silently dropped in the
            # pinned version (verified empirically); only the
            # constructor-bound ``system_prompt`` reaches the model. The
            # current policy prompt is therefore baked into ``agent`` by
            # ``_build_agent``, triggered from ``set_prompts``.
            async with agent.iter(
                question,
                deps=deps,
                usage_limits=UsageLimits(request_limit=self.max_iters + 1),
            ) as run:
                async for _ in run:
                    pass
                result = run.result
                if result is not None:
                    output = result.output
                    answer = str(getattr(output, "answer", "") or "").strip()
                    messages = list(result.all_messages())
        except Exception:
            logger.exception("HotpotQA PydanticAI agent failed for question %r", question[:80])

        tool_calls = _build_agent_tool_calls(
            messages=messages,
            deps=deps,
            final_answer=answer,
        )
        return HotpotQAAgentOutput(
            answer=answer,
            retrieved_paragraphs=list(deps.retrieved),
            tool_calls=tool_calls,
        )

    # ─── Tool implementations ────────────────────────────────────────────

    def _do_retrieve(self, deps: HotpotQADeps, query: str) -> str:
        remaining_before = remaining_gold_titles(deps.gold_supporting_titles, deps.retrieved_titles_seen)
        if deps.retrieve_k_fn is not None:
            # Runtime-injected retriever — currently distractor-over-bm25
            # or fullwiki-bm25s, dispatched by ``cfg.retrieval_mode`` in
            # ``build_pydantic_agent_runtime``. The injected fn dedupes
            # within a single call; apply the agent's cross-call dedup
            # post-hoc so already-retrieved titles aren't re-shown to
            # the agent.
            candidates = deps.retrieve_k_fn(query, deps.retrieve_k)
            hits = [p for p in candidates if p.title not in deps.retrieved_titles_seen]
        else:
            # Legacy local-context path (used by direct test callers
            # that don't go through the runtime). Filters candidates
            # against already-retrieved titles via ``bm25_top_k``.
            hits = bm25_top_k(
                query=query,
                paragraphs=deps.paragraphs,
                top_k=deps.retrieve_k,
                already_retrieved_titles=deps.retrieved_titles_seen,
            )
        for paragraph in hits:
            if paragraph.title not in deps.retrieved_titles_seen:
                deps.retrieved_titles_seen.add(paragraph.title)
                deps.retrieved.append(paragraph)
        remaining_after = remaining_gold_titles(deps.gold_supporting_titles, deps.retrieved_titles_seen)
        deps.tool_invocations.append(
            {
                "tool": "retrieve_k",
                "args": {"query": query},
                "retrieved_titles": [p.title for p in hits],
                "gold_titles_remaining_before": remaining_before,
                "gold_titles_remaining_after": remaining_after,
            }
        )
        if not hits:
            return "No new paragraphs matched the query."
        return "\n\n".join(f"### {p.title}\n{p.text}" for p in hits)

    async def _do_summarize(
        self,
        deps: HotpotQADeps,
        *,
        question: str,
        passages: list[str],
        context: str | None,
    ) -> str:
        user_prompt = _build_summarize_user_prompt(question=question, passages=passages, context=context)
        # Prefer the per-case snapshot ``forward`` captured at the
        # start of the run so a concurrent ``set_prompts`` can't
        # swap the prompt mid-tool-call. Fall back to instance state
        # for direct test callers that build ``HotpotQADeps`` by hand
        # without going through ``forward``.
        summarize_prompt = (
            deps.summarize_prompt_override
            if deps.summarize_prompt_override is not None
            else self._current_summarize_prompt
        )
        summary = await self._summarize_llm_call(summarize_prompt, user_prompt)
        deps.tool_invocations.append(
            {
                "tool": "summarize",
                "args": {
                    "question": question,
                    "num_passages": len(passages),
                    "has_context": context is not None,
                },
                "summary": summary,
                "gold_titles_remaining_before": remaining_gold_titles(
                    deps.gold_supporting_titles, deps.retrieved_titles_seen
                ),
                "gold_titles_remaining_after": remaining_gold_titles(
                    deps.gold_supporting_titles, deps.retrieved_titles_seen
                ),
            }
        )
        return summary

    async def _default_summarize_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Default summarize implementation: a direct OpenAI chat call.

        The optimizable ``system_prompt`` is right there in ``messages``
        — that's the whole point of bypassing a sub-Agent. The reader
        scanning this function can see exactly what GEPA optimizes and
        how it lands in the API call.
        """
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI()
        try:
            response = await self._openai_client.chat.completions.create(
                model=self._summarize_model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            logger.exception("HotpotQA summarize OpenAI call failed")
            return ""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        content = getattr(getattr(choices[0], "message", None), "content", None)
        return str(content or "").strip()


# ─── Trajectory harvesting ──────────────────────────────────────────────


def _build_agent_tool_calls(
    *,
    messages: Sequence[Any],
    deps: HotpotQADeps,
    final_answer: str,
) -> list[AgentToolCall]:
    """Walk PydanticAI's message stream into ``AgentToolCall`` steps.

    Pairs outgoing ``ToolCallPart`` instances with their corresponding
    ``ToolReturnPart`` (by ``tool_call_id``) and zips with the agent's
    own ``deps.tool_invocations`` log so each step carries the
    gold-title bookkeeping the trajectory dict doesn't know about.
    PydanticAI's structured-output termination surfaces as a
    ``final_result`` tool call carrying the ``HotpotQAOutput``; we map
    that to a synthetic ``finish`` step so the rilixai trajectory
    schema is uniform.
    """
    tool_returns_by_id: dict[str, Any] = {}
    for message in messages:
        for part in getattr(message, "parts", None) or []:
            if type(part).__name__ == "ToolReturnPart":
                tool_call_id = getattr(part, "tool_call_id", None)
                if tool_call_id is not None:
                    tool_returns_by_id[str(tool_call_id)] = part

    invocations_by_tool: dict[str, list[dict[str, Any]]] = {}
    for invocation in deps.tool_invocations:
        invocations_by_tool.setdefault(invocation["tool"], []).append(invocation)
    cursors: dict[str, int] = dict.fromkeys(invocations_by_tool, 0)

    steps: list[AgentToolCall] = []
    step_index = 0
    pending_thought = ""
    saw_final_result = False

    for message in messages:
        text_buffer: list[str] = []
        for part in getattr(message, "parts", None) or []:
            part_type = type(part).__name__
            if part_type == "TextPart":
                text_buffer.append(str(getattr(part, "content", "") or ""))
            elif part_type == "ToolCallPart":
                tool_name = str(getattr(part, "tool_name", "") or "")
                args_raw = getattr(part, "args", {}) or {}
                tool_args = args_raw if isinstance(args_raw, Mapping) else {"raw": str(args_raw)}
                tool_call_id = str(getattr(part, "tool_call_id", "") or "")
                observation_part = tool_returns_by_id.get(tool_call_id)
                observation = (
                    str(getattr(observation_part, "content", "") or "") if observation_part is not None else ""
                )
                thought = (
                    pending_thought + ("\n" if pending_thought and text_buffer else "") + "".join(text_buffer)
                ).strip()
                pending_thought = ""
                text_buffer = []

                if tool_name == "final_result":
                    saw_final_result = True
                    snapshot = remaining_gold_titles(deps.gold_supporting_titles, deps.retrieved_titles_seen)
                    steps.append(
                        AgentToolCall(
                            step_index=step_index,
                            tool_name="finish",
                            tool_args=dict(tool_args),
                            observation=observation,
                            thought=thought,
                            gold_titles_remaining_before=snapshot,
                            gold_titles_remaining_after=snapshot,
                        )
                    )
                    step_index += 1
                    continue

                invocation_list = invocations_by_tool.get(tool_name, [])
                cursor = cursors.get(tool_name, 0)
                if cursor < len(invocation_list):
                    invocation = invocation_list[cursor]
                    cursors[tool_name] = cursor + 1
                    remaining_before = list(invocation["gold_titles_remaining_before"])
                    remaining_after = list(invocation["gold_titles_remaining_after"])
                else:
                    snapshot = remaining_gold_titles(deps.gold_supporting_titles, deps.retrieved_titles_seen)
                    remaining_before = snapshot
                    remaining_after = snapshot

                steps.append(
                    AgentToolCall(
                        step_index=step_index,
                        tool_name=tool_name,
                        tool_args=dict(tool_args),
                        observation=observation,
                        thought=thought,
                        gold_titles_remaining_before=remaining_before,
                        gold_titles_remaining_after=remaining_after,
                    )
                )
                step_index += 1
        if text_buffer:
            pending_thought = ("\n".join(text_buffer)).strip()

    if final_answer and not saw_final_result:
        snapshot = remaining_gold_titles(deps.gold_supporting_titles, deps.retrieved_titles_seen)
        steps.append(
            AgentToolCall(
                step_index=step_index,
                tool_name="finish",
                tool_args={"answer": final_answer},
                observation="",
                thought=pending_thought,
                gold_titles_remaining_before=snapshot,
                gold_titles_remaining_after=snapshot,
            )
        )
    return steps


# ─── Helpers ─────────────────────────────────────────────────────────────


def _build_summarize_user_prompt(
    *,
    question: str,
    passages: list[str],
    context: str | None,
) -> str:
    parts: list[str] = [f"Question: {question}"]
    if context:
        parts.append(f"Prior context:\n{context}")
    parts.append("Passages (one per block):\n\n" + "\n\n".join(passages))
    parts.append("Produce a concise summary relevant to the question.")
    return "\n\n".join(parts)
