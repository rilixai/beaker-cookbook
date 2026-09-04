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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiers.clients import Client
from verifiers.types import RolloutInput

from automationbench_skills.data.tasks import Sample
from automationbench_skills.prompts import load_system_prompt, with_system_prompt
from automationbench_skills.skills_tools import SKILL_TOOLS, set_skills_dir
from automationbench_skills.vendored.model_setup import (
    build_client,
    build_sampling_args,
    resolve_api,
    resolve_api_key_var,
)


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "xhigh"
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
    chat/responses; gateway models via base_url). Not a capability profile."""

    name: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key_var: str = "OPENAI_API_KEY"
    api: str = "auto"  # or: anthropic | chat_completions | responses | gemini_interactions
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    extra_body: str | None = None  # raw JSON merged into every request body

    def resolved_api(self) -> str:
        return str(resolve_api(self.name, self.base_url, self.api))


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
        _ENV_CACHE[key] = AutomationBenchEnv(
            dataset=dataset,
            rubric=create_rubric(),
            tools=list(SKILL_TOOLS) if skills else None,
            max_turns=max_steps,
            toolset=toolset,
            timeout_seconds=timeout,
        )
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
        key_var = resolve_api_key_var(resolved, model.api_key_var)
        _CLIENT_CACHE[key] = build_client(resolved, key_var, model.base_url)
    return _CLIENT_CACHE[key]


def _rollout_input(sample: Sample, system_prompt: str | None = None) -> RolloutInput:
    return RolloutInput(
        prompt=with_system_prompt(sample.prompt, system_prompt),
        example_id=sample.index,
        answer=sample.answer,
        info=sample.info,
    )


def _to_result(sample: Sample, output: dict[str, Any]) -> RunResult:
    metrics = output.get("metrics") or {}
    partial = float(metrics.get("partial_credit", output.get("reward", 0.0)))
    strict = float(metrics.get("task_completed_correctly", 1.0 if partial == 1.0 else 0.0))
    completion = output.get("completion") or []
    trajectory = [m if isinstance(m, dict) else m.model_dump(mode="json") for m in completion]
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
    )


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
    env = get_env(toolset=toolset, skills=skills_dir is not None, max_steps=max_steps, timeout=timeout)
    set_skills_dir(skills_dir)
    client = get_client(model)
    sampling_args = build_sampling_args(model.name, model.resolved_api(), model.reasoning_effort, model.extra_body)
    rollout = env.run_rollout(
        _rollout_input(sample, load_system_prompt(prompts_dir, skills=skills_dir is not None)),
        client,
        model.name,
        sampling_args or {},
        state_columns=STATE_COLUMNS,
    )
    try:
        output = await (asyncio.wait_for(rollout, timeout + TIMEOUT_GRACE_SECONDS) if timeout else rollout)
    except TimeoutError:
        return RunResult(
            task_name=sample.task_name,
            domain=sample.domain,
            partial_credit=0.0,
            task_completed_correctly=0.0,
            trajectory=[],
            end_state=None,
            error=f"timeout after {timeout}s",
        )
    return _to_result(sample, output)


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
