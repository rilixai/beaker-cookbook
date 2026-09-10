"""The baseline OfficeQA agent loop.

Constants come from the OfficeQA Pro report (arXiv:2603.08655) §4.1 and are
defined in :mod:`officeqa.config` with their source comments. The model
client is injectable so tests can drive a full episode with a scripted
client and no network.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from officeqa import config
from officeqa.agent import prompts
from officeqa.agent.tools import ToolSet
from officeqa.data.dataset import AgentInput


logger = logging.getLogger(__name__)

_FINAL_ANSWER_RE = re.compile(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE)


def extract_final_answer(text: str | None) -> str | None:
    """Last ``<FINAL_ANSWER>...</FINAL_ANSWER>`` body, or ``None`` (which scores 0)."""
    if not text:
        return None
    matches = _FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    return str(matches[-1]).strip()


# --- Model client -----------------------------------------------------------


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.reasoning_tokens += other.reasoning_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """One assistant turn in OpenAI chat format: ``content`` and/or ``tool_calls``."""

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


def price_for(model: str) -> tuple[float, float, float] | None:
    """(input, output, cached) USD per MTok from our table, matching on the bare model name."""
    bare = model.split("/")[-1]
    return config.MODEL_PRICING_USD_PER_MTOK.get(bare)


def estimate_cost(model: str, usage: Usage) -> float | None:
    p = price_for(model)
    if p is None:
        return None
    inp, out, cached = p
    uncached = max(0, usage.prompt_tokens - usage.cached_tokens)
    return (uncached * inp + usage.cached_tokens * cached + usage.completion_tokens * out) / 1e6


class LiteLLMClient:
    """LiteLLM-backed client; ``reasoning_effort`` is passed explicitly (report App. D.2)."""

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = config.DEFAULT_REASONING_EFFORT,
        timeout_s: float = config.LLM_TIMEOUT_S,
        max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
        num_retries: int = 2,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.num_retries = num_retries

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        import litellm

        litellm.suppress_debug_info = True
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout_s,
            "num_retries": self.num_retries,
            "max_tokens": self.max_output_tokens,
            "drop_params": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_effort and self.reasoning_effort != "none":
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = await litellm.acompletion(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[dict[str, Any]] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
            )
        usage = Usage()
        u = getattr(resp, "usage", None)
        if u is not None:
            usage.prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
            usage.completion_tokens = int(getattr(u, "completion_tokens", 0) or 0)
            ptd = getattr(u, "prompt_tokens_details", None)
            usage.cached_tokens = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd is not None else 0
            ctd = getattr(u, "completion_tokens_details", None)
            usage.reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd is not None else 0
        cost: float | None
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            cost = estimate_cost(self.model, usage)
        return LLMResponse(content=msg.content, tool_calls=tool_calls, usage=usage, cost_usd=cost)


# --- Episode ----------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    usage: Usage
    cost_usd: float | None
    latency_s: float
    tool_calls: int


@dataclass
class Episode:
    """Everything the loop produced for one rollout. Mutated in place so a
    timeout still leaves a partial trajectory behind."""

    trajectory: list[dict[str, Any]] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    final_answer: str | None = None
    status: str = "running"  # answered | no_answer | running
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    cost_known: bool = True
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    truncated_tool_outputs: int = 0

    @property
    def tool_calls(self) -> int:
        return sum(self.tool_call_counts.values())

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_answer": self.final_answer,
            "steps": self.n_steps,
            "tool_calls": self.tool_calls,
            "tool_call_counts": dict(self.tool_call_counts),
            "truncated_tool_outputs": self.truncated_tool_outputs,
            "usage": asdict(self.usage),
            "cost_usd": self.cost_usd if self.cost_known else None,
            "trajectory": self.trajectory,
            "step_records": [asdict(s) for s in self.steps],
        }


def truncate_output(text: str, limit: int) -> tuple[str, bool]:
    """Cap a tool result at ``limit`` characters (report §4.1 fn. 9), keeping head and tail."""
    if len(text) <= limit:
        return text, False
    marker = f"\n\n... [truncated {len(text) - limit:,} characters; output capped at {limit:,}] ...\n\n"
    keep = max(0, limit - len(marker))
    head = int(keep * 0.7)
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def window_messages(history: Sequence[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    """Return the messages actually sent to the model for this step.

    Report §4.1: "a sliding window that retains the 30 most recent messages".
    Interpretation (ours): the system prompt and the initial user turn (the
    question) are pinned; the window applies to everything after them. The
    cut never lands between an assistant tool-call message and its tool
    results, since providers reject orphaned ``tool`` messages.
    """
    pinned = list(history[:2])
    rest = list(history[2:])
    if len(rest) > window_size:
        rest = rest[-window_size:]
        while rest and rest[0].get("role") == "tool":
            rest.pop(0)
    return pinned + rest


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"tool arguments are not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


class OfficeQAAgent:
    def __init__(
        self,
        client: LLMClient,
        toolset: ToolSet,
        *,
        max_steps: int = config.MAX_STEPS,
        window_size: int = config.WINDOW_SIZE,
        tool_output_limit: int = config.TOOL_OUTPUT_LIMIT,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._tools = toolset
        self._max_steps = max_steps
        self._window_size = window_size
        self._limit = tool_output_limit
        self._system_prompt = (
            system_prompt if system_prompt is not None else prompts.system_prompt_for(tuple(toolset.tools))
        )

    async def forward(self, sample: AgentInput, episode: Episode | None = None) -> Episode:
        ep = episode if episode is not None else Episode()
        ep.trajectory.append({"role": "system", "content": self._system_prompt})
        ep.trajectory.append({"role": "user", "content": prompts.user_message(json.dumps(sample.to_json(), indent=2))})
        schemas = self._tools.schemas()

        for step in range(1, self._max_steps + 1):
            remaining = self._max_steps - step + 1
            messages = window_messages(ep.trajectory, self._window_size)
            messages.append({"role": "user", "content": prompts.step_reminder(remaining)})

            t0 = time.monotonic()
            resp = await self._client.complete(messages, schemas)
            latency = time.monotonic() - t0
            ep.usage.add(resp.usage)
            if resp.cost_usd is None:
                ep.cost_known = False
            else:
                ep.cost_usd += resp.cost_usd
            ep.steps.append(
                StepRecord(
                    step=step,
                    usage=resp.usage,
                    cost_usd=resp.cost_usd,
                    latency_s=latency,
                    tool_calls=len(resp.tool_calls),
                )
            )
            ep.trajectory.append(resp.as_message())

            if resp.tool_calls:
                for tc in resp.tool_calls:
                    name = tc["function"]["name"]
                    ep.tool_call_counts[name] = ep.tool_call_counts.get(name, 0) + 1
                    try:
                        args = _parse_arguments(tc["function"]["arguments"])
                    except ValueError as e:
                        output = f"Error: {e}"
                    else:
                        output = await self._tools.call(name, args)
                    output, truncated = truncate_output(output, self._limit)
                    if truncated:
                        ep.truncated_tool_outputs += 1
                    ep.trajectory.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": output})
                continue

            answer = extract_final_answer(resp.content)
            if answer is not None:
                ep.final_answer = answer
                ep.status = "answered"
                return ep
            if remaining > 1:
                ep.trajectory.append(
                    {
                        "role": "user",
                        "content": "No <FINAL_ANSWER> tag found. Continue working, or give your final answer in the required format.",
                    }
                )

        ep.status = "no_answer"
        return ep
