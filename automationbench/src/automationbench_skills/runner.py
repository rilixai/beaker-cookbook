"""The core export: a single agent inference on a single task, as a function.

``run_one(sample, model=..., skills_dir=..., prompts_dir=...)`` executes ONE
verifiers rollout (``env.run_rollout`` — generate + deterministic rubric
scoring) and returns both metrics plus the trajectory and end-of-rollout world
state. The Beaker optimizer calls it in a loop: run a train sample, inspect
trajectory+score, edit files in ``skills_dir`` and ``prompts_dir``, repeat,
then run held-out test.

The ``AutomationBenchEnv`` is built once per (toolset, skills on/off,
max_turns) and reused across calls — its ``setup_state`` resets the per-task
world every rollout. Only the sample, the ``skills_dir`` contents and the
prompt file in ``prompts_dir`` vary per call; both are read live, so editing
them between calls changes agent behavior with no env rebuild.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiers.clients import Client
from verifiers.legacy.types import ClientConfig
from verifiers.types import RolloutInput

from automationbench_skills.clients import CostTrackingChatCompletionsClient
from automationbench_skills.data.tasks import Sample
from automationbench_skills.prompts import load_system_prompt, with_system_prompt
from automationbench_skills.skills_tools import SKILL_TOOLS, set_skills_dir
from automationbench_skills.vendored.model_setup import (
    build_client,
    build_sampling_args,
    resolve_api,
    resolve_api_key_var,
)


DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_REASONING_EFFORT = "max"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_VAR = "OPENROUTER_API_KEY"
DEFAULT_MAX_STEPS = 50  # upstream eval.py's --max-turns default
# Scoring and cleanup run after the env's own timeout stops the loop; the outer
# guard only catches a rollout stuck outside that loop.
TIMEOUT_GRACE_SECONDS = 60.0
# State fields upstream's eval exports alongside each rollout.
STATE_COLUMNS = ["_usage", "_debug", "_assertion_results", "_end_state", "_perf"]


@dataclass(frozen=True)
class ModelSpec:
    """A model selection threaded straight into AutomationBench's own routing
    (Anthropic-native for claude-*, Gemini interactions for gemini-*, OpenAI
    chat/responses; gateway models via base_url), plus OpenRouter for
    ``vendor/model`` names. Not a capability profile."""

    name: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key_var: str = "OPENAI_API_KEY"
    api: str = "auto"  # or: anthropic | chat_completions | responses | gemini_interactions
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    reasoning_enabled: bool | None = None
    extra_body: str | None = None  # raw JSON merged into every request body

    def is_openrouter(self) -> bool:
        return "openrouter.ai" in (self.base_url or "") or (self.base_url is None and "/" in self.name)

    def effective_base_url(self) -> str | None:
        if self.is_openrouter() and self.base_url is None:
            return OPENROUTER_BASE_URL
        return self.base_url

    def effective_api_key_var(self) -> str:
        if self.is_openrouter() and self.api_key_var == "OPENAI_API_KEY":
            return OPENROUTER_API_KEY_VAR
        return str(resolve_api_key_var(self.resolved_api(), self.api_key_var))

    def resolved_api(self) -> str:
        if self.is_openrouter() and self.api == "auto":
            return "chat_completions"
        return str(resolve_api(self.name, self.base_url, self.api))

    def sampling_args(self) -> dict[str, Any]:
        effort = self.reasoning_effort if self.reasoning_effort not in (None, "", "default") else None
        if self.is_openrouter():
            reasoning: dict[str, Any] = {}
            if effort is not None:
                reasoning["effort"] = effort
            if self.reasoning_enabled is not None:
                reasoning["enabled"] = self.reasoning_enabled
            extra_body: dict[str, Any] = {"reasoning": reasoning} if reasoning else {}
            if self.extra_body:
                extra_body = {**extra_body, **json.loads(self.extra_body)}
            return {"extra_body": extra_body} if extra_body else {}
        return build_sampling_args(self.name, self.resolved_api(), effort, self.extra_body) or {}


@dataclass
class RunResult:
    """Outcome of one rollout on one task."""

    task_name: str
    domain: str
    partial_credit: float  # fraction of scored assertions passed (0-1)
    task_completed_correctly: float  # strict 0/1 benchmark metric
    trajectory: list[dict[str, Any]]  # completion messages incl. tool calls/results
    end_state: dict[str, Any] | None  # final WorldState dump (debugging)
    assertion_results: list[dict[str, Any]] = field(default_factory=list)
    error: Any | None = None
    raw: dict[str, Any] = field(default_factory=dict)  # full verifiers RolloutOutput
    latency_s: float | None = None
    cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    perf: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "domain": self.domain,
            "partial_credit": self.partial_credit,
            "task_completed_correctly": self.task_completed_correctly,
            "trajectory": self.trajectory,
            "end_state": self.end_state,
            "assertion_results": self.assertion_results,
            "error": str(self.error) if self.error is not None else None,
            "latency_s": self.latency_s,
            "cost_usd": self.cost_usd,
            "usage": self.usage,
            "perf": self.perf,
        }


_ENV_CACHE: dict[tuple[str, bool, int, float | None], Any] = {}
_CLIENT_CACHE: dict[tuple[ModelSpec, Any], Client] = {}


def get_env(
    toolset: str = "zapier",
    skills: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: float | None = None,
) -> Any:
    """Build (once) and return the shared AutomationBenchEnv.

    The skill tools are registered via the env's ``tools=`` parameter; under
    the ``zapier`` (meta-tools) and ``api`` toolsets, added tools survive
    ``setup_state``'s per-task filtering. ``limited_zapier`` filters to each
    task's declared tool list and would drop them — so this recipe pins
    ``zapier``/``api`` and refuses ``limited_zapier`` when skills are on.

    ``timeout`` (seconds) bounds the rollout loop inside the env: the loop
    stops on expiry and the rubric still scores the world the agent has
    mutated so far.
    """
    if toolset == "limited_zapier" and skills:
        raise ValueError(
            "toolset='limited_zapier' filters tools to each task's declared list at "
            "setup_state, which drops the skill tools. Use toolset='zapier' (default)."
        )
    key = (toolset, skills, max_steps, timeout)
    if key not in _ENV_CACHE:
        import json

        from automationbench.rubric import create_rubric
        from automationbench.runner import AutomationBenchEnv
        from beaker.tracing.integrations import verifiers as beaker_verifiers
        from datasets import Dataset

        from automationbench_skills.data.tasks import load_samples

        # The env's own dataset only backs env.evaluate(); this recipe drives
        # rollouts explicitly through run_rollout, so a 1-row placeholder
        # satisfies the constructor without materializing 800 tasks into a
        # HuggingFace Dataset (whose schema normalization mangles task infos).
        first = load_samples()[0]
        dataset = Dataset.from_list(
            [
                {
                    "example_id": first.index,
                    "prompt": first.prompt,
                    "answer": first.answer,
                    "info": json.dumps(first.info),
                }
            ]
        )
        env = AutomationBenchEnv(
            dataset=dataset,
            rubric=create_rubric(),
            tools=list(SKILL_TOOLS) if skills else None,
            max_turns=max_steps,
            toolset=toolset,
            timeout_seconds=timeout,
        )
        # One ``tool_call`` span per tool execution, under whichever Beaker
        # capture is active at call time; the hidden ``world`` arg stays out.
        _ENV_CACHE[key] = beaker_verifiers.instrument(env)
    return _ENV_CACHE[key]


def get_client(model: ModelSpec) -> Client:
    """Return the client for ``model`` on the current event loop.

    The cache is keyed by (model, running loop): the async client's connection
    pool binds to the loop it first runs under, so a client built inside one
    ``asyncio.run`` cannot be reused inside the next. Repeated ``run_one``
    calls therefore get a fresh client per loop, while rollouts sharing a loop
    (e.g. ``run_split``) share one client and its connection pool.
    """
    for stale in [k for k in _CLIENT_CACHE if k[1].is_closed()]:
        del _CLIENT_CACHE[stale]
    key = (model, asyncio.get_running_loop())
    if key not in _CLIENT_CACHE:
        resolved = model.resolved_api()
        key_var = model.effective_api_key_var()
        if not os.environ.get(key_var):
            raise ValueError(f"No API key found. Set the {key_var} environment variable.")
        if resolved == "chat_completions":
            _CLIENT_CACHE[key] = CostTrackingChatCompletionsClient(
                ClientConfig(
                    api_key_var=key_var,
                    api_base_url=model.effective_base_url() or "https://api.openai.com/v1",
                    extra_headers={},
                )
            )
        else:
            _CLIENT_CACHE[key] = build_client(resolved, key_var, model.effective_base_url())
    return _CLIENT_CACHE[key]


def _rollout_input(sample: Sample, system_prompt: str | None = None) -> RolloutInput:
    return RolloutInput(
        prompt=with_system_prompt(sample.prompt, system_prompt),
        example_id=sample.index,
        answer=sample.answer,
        info=sample.info,
    )


def _to_result(sample: Sample, output: dict[str, Any], latency_s: float | None = None) -> RunResult:
    metrics = output.get("metrics") or {}
    partial = float(metrics.get("partial_credit", output.get("reward", 0.0)))
    strict = float(metrics.get("task_completed_correctly", 1.0 if partial == 1.0 else 0.0))
    completion = output.get("completion") or []
    trajectory = [m if isinstance(m, dict) else m.model_dump(mode="json") for m in completion]
    usage = output.get("_usage") or {}
    perf = output.get("_perf") or {}
    return RunResult(
        task_name=sample.task_name,
        domain=sample.domain,
        partial_credit=partial,
        task_completed_correctly=strict,
        trajectory=trajectory,
        end_state=output.get("_end_state"),
        assertion_results=output.get("_assertion_results") or [],
        error=output.get("error"),
        raw=dict(output),
        latency_s=latency_s,
        cost_usd=perf.get("cost_usd"),
        usage=usage,
        perf=perf,
    )


async def run_rollout_raw(
    sample: Sample,
    *,
    model: ModelSpec,
    client: Client,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "zapier",
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run ONE agent rollout with ``client`` and return verifiers' raw output.

    The shared execution path behind :func:`run_one_async` and the Beaker
    integration: environment, skills directory, system prompt, sampling
    arguments and the outer watchdog are set up identically for both. Raises
    ``TimeoutError`` when the rollout outlives ``timeout`` plus its grace.
    """
    env = get_env(toolset=toolset, skills=skills_dir is not None, max_steps=max_steps, timeout=timeout)
    set_skills_dir(skills_dir)
    sampling_args = model.sampling_args()
    rollout = env.run_rollout(
        _rollout_input(sample, load_system_prompt(prompts_dir, skills=skills_dir is not None)),
        client,
        model.name,
        sampling_args or {},
        state_columns=STATE_COLUMNS,
    )
    output = await (asyncio.wait_for(rollout, timeout + TIMEOUT_GRACE_SECONDS) if timeout else rollout)
    return dict(output)


async def run_one_async(
    sample: Sample,
    *,
    model: ModelSpec | str = DEFAULT_MODEL,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "zapier",
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: float | None = None,
) -> RunResult:
    """Run ONE agent rollout on one task and score it with the benchmark rubric.

    Side-effect-free besides reading ``skills_dir`` and ``prompts_dir``: the
    task's simulated world is created fresh inside the rollout and returned in
    ``RunResult.end_state``. ``skills_dir=None`` is the baseline arm — the skill
    tools are absent. The agent's system prompt is ``prompts_dir/system.md``
    (``system_no_skills.md`` in the baseline arm); with ``None`` (or no such
    file) the dataset row's system message is used.
    ``timeout`` (seconds) bounds the rollout: on expiry the loop stops and the
    partially mutated world is still scored, so completed steps keep their
    partial credit.
    """
    if isinstance(model, str):
        model = ModelSpec(name=model)
    started = time.monotonic()
    try:
        output = await run_rollout_raw(
            sample,
            model=model,
            client=get_client(model),
            skills_dir=skills_dir,
            prompts_dir=prompts_dir,
            toolset=toolset,
            max_steps=max_steps,
            timeout=timeout,
        )
    except TimeoutError:
        return RunResult(
            task_name=sample.task_name,
            domain=sample.domain,
            partial_credit=0.0,
            task_completed_correctly=0.0,
            trajectory=[],
            end_state=None,
            error=f"timeout after {timeout}s",
            latency_s=time.monotonic() - started,
        )
    return _to_result(sample, output, latency_s=time.monotonic() - started)


def run_one(
    sample: Sample,
    *,
    model: ModelSpec | str = DEFAULT_MODEL,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "zapier",
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: float | None = None,
) -> RunResult:
    """Synchronous wrapper around :func:`run_one_async`."""
    return asyncio.run(
        run_one_async(
            sample,
            model=model,
            skills_dir=skills_dir,
            prompts_dir=prompts_dir,
            toolset=toolset,
            max_steps=max_steps,
            timeout=timeout,
        )
    )


async def run_split_async(
    samples: list[Sample],
    *,
    model: ModelSpec | str = DEFAULT_MODEL,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "zapier",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_concurrent: int = 8,
    timeout: float | None = None,
    on_result: Any | None = None,
) -> list[RunResult]:
    """Thin concurrency wrapper over :func:`run_one_async` (one shared skills_dir/prompts_dir)."""
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(sample: Sample) -> RunResult:
        async with sem:
            result = await run_one_async(
                sample,
                model=model,
                skills_dir=skills_dir,
                prompts_dir=prompts_dir,
                toolset=toolset,
                max_steps=max_steps,
                timeout=timeout,
            )
        if on_result is not None:
            on_result(result)
        return result

    return list(await asyncio.gather(*(bounded(s) for s in samples)))


def run_split(
    samples: list[Sample],
    *,
    model: ModelSpec | str = DEFAULT_MODEL,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "zapier",
    max_steps: int = DEFAULT_MAX_STEPS,
    max_concurrent: int = 8,
    timeout: float | None = None,
    on_result: Any | None = None,
) -> list[RunResult]:
    """Synchronous wrapper around :func:`run_split_async`."""
    return asyncio.run(
        run_split_async(
            samples,
            model=model,
            skills_dir=skills_dir,
            prompts_dir=prompts_dir,
            toolset=toolset,
            max_steps=max_steps,
            max_concurrent=max_concurrent,
            timeout=timeout,
            on_result=on_result,
        )
    )
