"""Run the PI-native task and NullHarness through an OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from automationbench.rubric import partial_credit
from automationbench.runner import AutomationBenchTaskConfig, AutomationBenchToolsetConfig, compute_connected_services
from automationbench.schema.world import WorldState
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from verifiers.v1.types import Usage

from automationbench_skills.clients import (
    DEFAULT_MAX_ANSWER_TOKENS,
    DEFAULT_MAX_REASONING_TOKENS,
    TracedInterception,
)
from automationbench_skills.data.tasks import Sample
from automationbench_skills.environment import BeakerData, BeakerTask
from automationbench_skills.grading_state import assertion_outcomes
from automationbench_skills.prompts import load_system_prompt, task_clock, with_system_prompt
from automationbench_skills.reference_context import reference_context


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_VAR = "OPENROUTER_API_KEY"
DEFAULT_MAX_STEPS = 50  # upstream eval.py's --max-turns default


@dataclass(frozen=True)
class ModelSpec:
    """A model served through a Chat Completions-compatible endpoint."""

    name: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key_var: str = "OPENAI_API_KEY"
    api: str = "auto"
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    reasoning_enabled: bool | None = None
    extra_body: str | None = None

    def __post_init__(self) -> None:
        if self.api not in {"auto", "chat_completions"}:
            raise ValueError("PI's native harness requires Chat Completions; use a compatible gateway for this model")

    def is_openrouter(self) -> bool:
        return "openrouter.ai" in (self.base_url or "") or (self.base_url is None and "/" in self.name)

    def effective_base_url(self) -> str | None:
        if self.is_openrouter() and self.base_url is None:
            return OPENROUTER_BASE_URL
        return self.base_url

    def effective_api_key_var(self) -> str:
        if self.is_openrouter() and self.api_key_var == "OPENAI_API_KEY":
            return OPENROUTER_API_KEY_VAR
        return self.api_key_var

    def sampling_args(self) -> dict[str, Any]:
        args: dict[str, Any]
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
            args = {"extra_body": extra_body} if extra_body else {}
        else:
            args = {"reasoning_effort": effort} if effort else {}
            if self.extra_body:
                args["extra_body"] = json.loads(self.extra_body)
        extra = args.get("extra_body") or {}
        # Reasoning shares the completion cap. An explicit reasoning budget
        # (OpenRouter's reasoning.max_tokens) sets the allowance; otherwise the
        # default allowance applies.
        requested_reasoning = extra.get("reasoning")
        budget = requested_reasoning.get("max_tokens") if isinstance(requested_reasoning, dict) else None
        if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0):
            raise ValueError("reasoning.max_tokens must be a positive integer")
        cap = (budget or DEFAULT_MAX_REASONING_TOKENS) + DEFAULT_MAX_ANSWER_TOKENS
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            for container in (args, extra):
                requested = container.pop(key, None)
                if requested is not None:
                    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
                        raise ValueError(f"{key} must be a positive integer")
                    cap = min(cap, requested)
        args["max_completion_tokens"] = cap
        return args


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
    raw: dict[str, Any] = field(default_factory=dict)  # full rollout output
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


def _rollout_input(sample: Sample, system_prompt: str | None = None) -> dict[str, Any]:
    return {
        "prompt": with_system_prompt(sample.prompt, system_prompt, clock=task_clock(sample.info)),
        "example_id": sample.index,
        "answer": sample.answer,
        "info": sample.info,
    }


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


def _subprocess_pythonpath() -> str:
    """Make tool-server imports resolve to the active source package."""
    source_dir = str(Path(__file__).resolve().parents[1])
    import_paths = [source_dir, *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(p for p in import_paths if p))


async def run_rollout_raw(
    sample: Sample,
    *,
    model: ModelSpec,
    api_key: str | None = None,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "api",
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run PI's native task/harness, retaining world evidence for trusted scoring."""
    # Tool servers start with `python -m` from a separate runtime directory.
    os.environ["PYTHONPATH"] = _subprocess_pythonpath()
    row = _rollout_input(sample, load_system_prompt(prompts_dir, skills=skills_dir is not None))
    info = row["info"]
    messages = row["prompt"]
    connected_services = compute_connected_services(info["initial_state"], info["assertions"], info["zapier_tools"])
    task_prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    if toolset == "api":
        task_prompt += reference_context(task_prompt, info["initial_state"], set(connected_services))
    data = BeakerData(
        idx=sample.index,
        name=sample.task_name,
        prompt=task_prompt,
        system_prompt="\n\n".join(m["content"] for m in messages if m["role"] == "system") or None,
        domain=sample.domain,
        assertions=info["assertions"],
        initial_state=info["initial_state"],
        connected_services=connected_services,
        skills_dir=str(Path(skills_dir).resolve()) if skills_dir is not None else None,
        scenario=info.get("_case_scenario", {}),
    )
    task = BeakerTask(
        data,
        AutomationBenchTaskConfig(
            tools=AutomationBenchToolsetConfig(
                mode=toolset, allowed_tools=info["zapier_tools"] if toolset == "limited_zapier" else []
            )
        ),
    )
    config = vf.AgentConfig(
        model=model.name,
        harness=NullHarnessConfig(id="null"),
        runtime=vf.SubprocessConfig(),
        client=vf.EvalClientConfig(
            base_url=model.effective_base_url() or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            api_key_var=model.effective_api_key_var(),
        ),
        sampling=vf.SamplingConfig(**model.sampling_args()),
        max_turns=max_steps,
        timeout=TimeoutConfig(setup=120, rollout=timeout, finalize=30, scoring=30),
    )
    try:
        async with TracedInterception(api_key) as interception:
            trace = await vf.Agent(config, interception=interception).run(task)
    finally:
        task.close_spans()
    artifacts = trace.state.artifacts
    initial = info["initial_state"]
    end_state = json.loads(artifacts["world.json"]) if "world.json" in artifacts else initial
    grading_state = json.loads(artifacts["grading-state.json"]) if "grading-state.json" in artifacts else None
    world = WorldState(**end_state)
    from automationbench_skills.grading_state import restore_grading_state

    restore_grading_state(world, grading_state)
    score = float(partial_credit({"info": info, "world": world, "initial_state": initial}))
    usage = Usage.aggregate(value for value in (trace.usage, interception.rejected_usage) if value is not None)
    return {
        "metrics": {"partial_credit": score, "task_completed_correctly": float(score == 1.0)},
        "reward": score,
        "completion": [m.model_dump(mode="json") for m in trace.messages if m.role in {"assistant", "tool"}],
        "error": trace.last_error,
        "stop_condition": "timeout_reached" if trace.is_timeout else trace.stop_condition,
        "_end_state": end_state,
        "_grading_state": grading_state,
        "_assertion_results": assertion_outcomes(world, info["assertions"], WorldState(**initial)),
        "_usage": {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.completion_tokens} if usage else {},
        "_perf": {"cost_usd": usage.cost if usage else None},
    }


async def run_one_async(
    sample: Sample,
    *,
    model: ModelSpec | str = DEFAULT_MODEL,
    skills_dir: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    toolset: str = "api",
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
    toolset: str = "api",
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
    toolset: str = "api",
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
    toolset: str = "api",
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
