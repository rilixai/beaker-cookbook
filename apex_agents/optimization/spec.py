"""Factory that assembles a :class:`PromptOptimizationSpec` for APEX-Agents.

Once built, the same ``run_optimization_from_spec`` and
``build_adapter_from_spec`` helpers used for production extraction
tasks run APEX-Agents end-to-end. The factory accepts an
:class:`ApexAgentsConfig` and the three-component seed candidate
(``system_prompt`` + ``task_template`` + ``resum_summary_prompt``).

``world_factory`` is plumbed through so tests can inject a
:class:`FakeWorld` factory; ``judge`` lets tests inject a stub rubric
judge. In production the defaults are the HF world builder + the
litellm-backed judge.

This module also exports the rilixai Modal-sandbox entry point
:func:`build_spec` — a ``@spec(name="apex-agents")``-decorated factory
that rilixai's sandbox dispatcher invokes once per remote optimization
run. See the cookbook README for the push + promote + trigger flow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rilixai import spec
from rilixai.prompt_optimization.models import Case, PromptCandidate
from rilixai.prompt_optimization.protocols import EvaluationProfile
from rilixai.prompt_optimization.spec import OptimizationContext, PromptOptimizationSpec

from ..agent.prompts import apex_agents_seed_candidate
from ..config import ApexAgentsConfig
from ..data.dataset import load_apex_agents_cases
from ..data.kfold import stratified_case_cap, world_held_out_val_split
from .metrics import (
    APEX_AGENTS_FIELD_WEIGHTS,
    ApexAgentsMetricsCalculator,
    build_apex_agents_field_extractor,
)
from .runtime import build_apex_agents_runtime


_APEX_AGENTS_PROFILE_KEY = "apex_agents"


def _apex_agents_agent_resolver(**_: Any) -> tuple[None, None]:
    """The runtime owns the agent + world, so adapter agent resolution is a no-op."""
    return (None, None)


def _build_apex_agents_profile_resolver(
    metrics: ApexAgentsMetricsCalculator,
    field_weights: dict[str, float] | None = None,
) -> Any:
    profile = EvaluationProfile(
        profile_key=_APEX_AGENTS_PROFILE_KEY,
        metrics_calculator=metrics,
        field_weights=dict(field_weights or APEX_AGENTS_FIELD_WEIGHTS),
    )

    def _resolver(**_: Any) -> EvaluationProfile:
        return profile

    return _resolver


def build_apex_agents_spec(
    *,
    cases_by_split: dict[str, Sequence[Case]],
    seed_candidate: PromptCandidate | None = None,
    config: ApexAgentsConfig | None = None,
    agent: Any | None = None,
    world_factory: Callable[[Any], Any] | None = None,
    model_factory: Callable[[str, float], Any] | None = None,
    judge: Callable[[str, str, str], bool] | None = None,
    name: str = "apex_agents",
    user_id: str = "__apex_agents_benchmark__",
    model: str | None = None,
    max_concurrency: int = 4,
    reflection_evidence_mode: str = "curated_plus_trace",
    field_weights: dict[str, float] | None = None,
) -> PromptOptimizationSpec:
    """Build a ready-to-run :class:`PromptOptimizationSpec` for APEX-Agents.

    ``world_factory`` is required at run time unless a pre-built
    ``agent`` is supplied — production code passes the HF world
    builder; tests pass a closure yielding :class:`FakeWorld`.
    ``judge`` defaults to the litellm rubric judge; tests inject a
    stub.

    ``curated_plus_trace`` is the default reflection mode because the
    runtime populates ``trace_evidence.per_component_feedback`` that
    scalar field scores cannot represent.
    """
    cfg = config or ApexAgentsConfig()
    seed = seed_candidate or apex_agents_seed_candidate()
    metrics = ApexAgentsMetricsCalculator()
    runtime = build_apex_agents_runtime(
        config=cfg,
        agent=agent,
        world_factory=world_factory,
        model_factory=model_factory,
        judge=judge,
    )
    return PromptOptimizationSpec(
        cases_by_split=cases_by_split,
        seed_candidate=seed,
        extraction_runtime=runtime,
        agent_resolver=_apex_agents_agent_resolver,
        field_extractor=build_apex_agents_field_extractor(),
        evaluation_profile_resolver=_build_apex_agents_profile_resolver(metrics, field_weights),
        name=name,
        user_id=user_id,
        model=model,
        task_type="apex_agents_agent",
        max_concurrency=max_concurrency,
        reflection_evidence_mode=reflection_evidence_mode,
    )


# ─── rilixai Modal sandbox entry point ─────────────────────────────────


# Defaults the sandbox build_spec applies when ctx.config omits a key.
# See README for the full key reference (GEPA vs cookbook split).
_DEFAULT_SANDBOX_CONFIG: dict[str, Any] = {
    "domain": "law",
    "val_worlds": 2,
    "val_size": 20,
    "train_size": 25,
    "task_model": "openai/gpt-4.1-mini-2025-04-14",
    "task_temperature": 0.0,
    "judge_model": "gemini/gemini-2.5-flash",
    "max_steps": 60,
    "cost_limit": 3.0,
    "seed": 0,
    "max_concurrency": 4,
}


@spec(
    name="apex-agents",
    description="APEX-Agents (Mercor law/IB) — faithful ReAct toolbelt agent + GEPA",
    metadata={"benchmark": "apex_agents", "agent_kind": "react_toolbelt"},
)
def build_spec(ctx: OptimizationContext) -> PromptOptimizationSpec:
    """Spec factory for the rilixai Modal sandbox path. See README for usage.

    No ``version=...`` argument here: ``sandbox.py --build`` supplies
    the push-time version (defaulting to ``v<short_sha>``) via
    ``rilixai push --version`` and then promotes the freshly-pushed
    row to ``apex-agents@production``. Trigger calls reference
    ``apex-agents@production`` so rilixai resolves the current
    promoted version server-side — no rilix-side / cookbook-side
    version bumps for routine deploys.
    """
    cfg_in: dict[str, Any] = {**_DEFAULT_SANDBOX_CONFIG, **dict(ctx.config or {})}
    apex_cfg = ApexAgentsConfig(
        task_model=str(cfg_in["task_model"]),
        task_temperature=float(cfg_in["task_temperature"]),
        judge_model=str(cfg_in["judge_model"]),
        max_steps=int(cfg_in["max_steps"]),
        cost_limit=float(cfg_in["cost_limit"]),
    )

    # Load all cases for the requested domain, then carve train/val
    # by world. Held-out val worlds are disjoint from train worlds,
    # so GEPA selects for cross-world transfer (not in-world fit) —
    # this avoids the val→test collapse the Law fold-0 run originally
    # showed.
    all_cases = load_apex_agents_cases(domain=str(cfg_in["domain"]))
    inner_train, validation = world_held_out_val_split(
        all_cases,
        n_val_worlds=int(cfg_in["val_worlds"]),
        seed=int(cfg_in["seed"]),
    )
    # Stratified caps: round-robin across worlds so train width stays
    # constant per ``train_size`` point and val sampling is balanced.
    train_cases = stratified_case_cap(inner_train, int(cfg_in["train_size"]), seed=int(cfg_in["seed"]))
    val_cases = stratified_case_cap(validation, int(cfg_in["val_size"]), seed=int(cfg_in["seed"]))

    # World factory: lazy HF download per case. Network is available
    # inside the Modal container; ``build_world_factory`` extracts the
    # per-case world zip + per-task input files on demand.
    from ..agent.world.world import build_world_factory

    return build_apex_agents_spec(
        cases_by_split={
            "train": list(train_cases),
            "validation": list(val_cases),
        },
        config=apex_cfg,
        world_factory=build_world_factory(),
        model=cfg_in.get("model") or ctx.model,
        max_concurrency=int(cfg_in["max_concurrency"]),
    )


__all__ = ["build_apex_agents_spec", "build_spec"]
