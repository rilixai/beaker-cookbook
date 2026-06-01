"""The whole APEX-Agents → rilixai integration: one ``@spec`` runner class.

``rilixai push`` targets this file. :class:`ApexAgentsRunner` is the entire
sandbox integration — rilixai assembles the metrics calculator, seed
candidate, and per-component feedback from the ``@spec`` declarations and the
runner's ``_package_result`` (which runs the rubric judge + emits the trace).
The ``@spec`` decorator builds the :class:`PromptOptimizationSpec` from the
runner class; rilixai resolves it via ``load_spec_from_target``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from rilixai import spec
from rilixai.adapters import BaseCaseRunner, CallableApplier, CaseRunResult
from rilixai.metrics import BaseMetricsCalculator, FieldConfig
from rilixai.prompt_optimization.models import Case

from .agent.prompts import apex_agents_seed_candidate
from .agent.types import ApexAgentsAgentOutput
from .config import ApexAgentsConfig
from .data.dataset import _APEX_AGENTS_GROUND_TRUTH_KEY, ApexAgentsRecord, load_apex_agents_cases
from .data.world_splits import stratified_case_cap, world_held_out_val_split
from .feedback import ApexAgentsFeedback
from .metrics import (
    RUBRIC_FIELD,
    build_apex_agents_run_metrics,
    build_rubric_judge,
    coerce_pass_rate,
    score_rubric,
)


@dataclass
class _ApexResult:
    """Per-case result: the precomputed rubric pass rate + the agent output.

    ``rubric_pass_rate`` is what the metrics calculator scores; ``agent_output``
    is what the feedback + trace builder read.
    """

    rubric_pass_rate: float
    final_answer: str
    agent_output: ApexAgentsAgentOutput


# ─── Scoring (judge runs in the runner; the metric reads the result) ─────


class ApexAgentsMetrics(BaseMetricsCalculator):
    """Single field: the judge-computed ``rubric_pass_rate`` the runner stashes.

    The comparator ignores ground truth and reads the precomputed float off the
    result; ``_has_valid_sample_for_comparison`` skips cases whose ground-truth
    bundle carries no rubric.
    """

    fields = [
        FieldConfig(
            name=RUBRIC_FIELD,
            extract_from=RUBRIC_FIELD,
            compare_to=_APEX_AGENTS_GROUND_TRUTH_KEY,
            comparators="rubric_pass_rate",
            weight=1.0,
        ),
    ]
    comparators = {"rubric_pass_rate": lambda predicted, _expected: coerce_pass_rate(predicted)}

    def _has_valid_sample_for_comparison(self, predicted: Any, actual: Any, cfg: Any) -> bool:
        return isinstance(actual, Mapping) and bool(actual.get("rubric"))


# ─── Sandbox run config ─────────────────────────────────────────────────


class ApexAgentsSandboxConfig(BaseModel):
    """Typed sandbox run config. rilixai validates ``ctx.config`` against this."""

    domain: str = "law"
    val_worlds: int = 2
    val_size: int = 20
    train_size: int = 25
    task_model: str = "openai/gpt-4.1-mini-2025-04-14"
    task_temperature: float = 0.0
    judge_model: str = "gemini/gemini-2.5-flash"
    max_steps: int = 60
    cost_limit: float = 3.0
    seed: int = 0
    max_concurrency: int = 4


def _apex_config_from(sandbox_cfg: ApexAgentsSandboxConfig) -> ApexAgentsConfig:
    return ApexAgentsConfig(
        task_model=sandbox_cfg.task_model,
        task_temperature=sandbox_cfg.task_temperature,
        judge_model=sandbox_cfg.judge_model,
        max_steps=sandbox_cfg.max_steps,
        cost_limit=sandbox_cfg.cost_limit,
    )


# ─── The integration: one @spec runner class ────────────────────────────


@spec(
    name="apex-agents",
    description="APEX-Agents (Mercor law/IB) — faithful ReAct toolbelt agent + GEPA",
    metadata={"benchmark": "apex_agents", "agent_kind": "react_toolbelt"},
    task_type="apex_agent",
    config_schema=ApexAgentsSandboxConfig,
    field_configs=ApexAgentsMetrics,
    feedback=ApexAgentsFeedback,
    # No explicit seed: rilixai auto-reads it from the agent's current
    # prompts via the applier's read() at spec-build time.
    # reflection_evidence_mode is kept (rilixai's default is "curated");
    # this agent emits rich trace_evidence the reflection LM should use.
    reflection_evidence_mode="curated_plus_trace",
    # rilixai's default max_concurrency is 8; 4 is the cost-bounded demo value.
    max_concurrency=4,
)
class ApexAgentsRunner(BaseCaseRunner[ApexAgentsRecord, _ApexResult]):
    """The entire APEX-Agents integration: one runner the rilixai sandbox drives."""

    def __init__(self, ctx: Any) -> None:
        sandbox_cfg = ctx.config if isinstance(ctx.config, ApexAgentsSandboxConfig) else ApexAgentsSandboxConfig()
        self._sandbox_cfg = sandbox_cfg
        cfg = _apex_config_from(sandbox_cfg)
        # Offline-test injection seam: a stub context may carry pre-built deps in
        # ``metadata`` (agent / world_factory / model_factory / judge) so the
        # runner constructs without network — the HF world download + the live
        # rubric judge. A production sandbox context carries none of these, so the
        # real factory + judge are built lazily here.
        injected: Mapping[str, Any] = getattr(ctx, "metadata", None) or {}
        agent = injected.get("agent")
        if agent is None:
            world_factory = injected.get("world_factory")
            if world_factory is None:
                from .agent.world.world import build_world_factory

                world_factory = build_world_factory()
            agent = _build_agent(cfg, world_factory=world_factory, model_factory=injected.get("model_factory"))
        judge = injected.get("judge") or build_rubric_judge(model=cfg.judge_model, timeout=cfg.llm_timeout)
        self._setup(cfg, agent, judge)

    def _setup(self, cfg: ApexAgentsConfig, agent: Any, judge: Callable[[str, str, str], bool]) -> None:
        self.cfg = cfg
        self.agent = agent
        self.judge = judge
        super().__init__(
            applier=CallableApplier(
                apply=agent.apply_candidate,
                read=lambda: dict(apex_agents_seed_candidate().components),
            )
        )

    async def run_case(self, record: ApexAgentsRecord) -> _ApexResult:
        output = await self.agent.forward(record=record)
        rubric_payload = [{"verifier_id": c.verifier_id, "criteria": c.criteria} for c in record.rubric]
        rubric_pass_rate = await asyncio.to_thread(
            score_rubric,
            rubric=rubric_payload,
            answer=output.final_answer,
            task_prompt=record.prompt,
            judge=self.judge,
        )
        return _ApexResult(rubric_pass_rate=rubric_pass_rate, final_answer=output.final_answer, agent_output=output)

    def _package_result(
        self,
        record: ApexAgentsRecord,
        output: _ApexResult,
        runtime_kwargs: Mapping[str, Any],
    ) -> CaseRunResult[_ApexResult]:
        run_metrics = build_apex_agents_run_metrics(
            record=record,
            output=output.agent_output,
            config=self.cfg,
            rubric_pass_rate=output.rubric_pass_rate,
        )
        # Per-component feedback flows through @spec(feedback=ApexAgentsFeedback);
        # the feedback methods read the agent output, not the result wrapper.
        feedback = self._build_feedback(record, output.agent_output, runtime_kwargs)  # type: ignore[arg-type]
        if feedback:
            run_metrics.setdefault("trace_evidence", {})["per_component_feedback"] = feedback
        return CaseRunResult(output=output, run_metrics=run_metrics)

    def cases_by_split(self, ctx: Any) -> dict[str, list[Case]]:
        sc = self._sandbox_cfg
        all_cases = load_apex_agents_cases(domain=sc.domain)
        inner_train, validation = world_held_out_val_split(all_cases, n_val_worlds=sc.val_worlds, seed=sc.seed)
        train_cases = stratified_case_cap(inner_train, sc.train_size, seed=sc.seed)
        val_cases = stratified_case_cap(validation, sc.val_size, seed=sc.seed)
        return {"train": list(train_cases), "validation": list(val_cases)}


def _build_agent(
    cfg: ApexAgentsConfig,
    *,
    world_factory: Callable[[Any], Any] | None,
    model_factory: Callable[[str, float], Any] | None,
) -> Any:
    from .agent.agent import ApexReActAgent
    from .agent.prompts import load_apex_agents_seed_prompts

    default_sys, default_task, default_resum = load_apex_agents_seed_prompts()
    return ApexReActAgent(
        model_name=cfg.task_model,
        model_temperature=cfg.task_temperature,
        max_steps=cfg.max_steps,
        cost_limit=cfg.cost_limit,
        max_toolbelt_size=cfg.max_toolbelt_size,
        max_context_tokens=cfg.max_context_tokens,
        default_system_prompt=default_sys,
        default_task_template=default_task,
        default_resum_summary_prompt=default_resum,
        world_factory=world_factory,  # type: ignore[arg-type]
        model_factory=model_factory,
        llm_timeout=cfg.llm_timeout,
    )
