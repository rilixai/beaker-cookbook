"""Factory that assembles a rilixai :class:`~rilixai.Spec` for APEX-Agents.

The SDK :class:`~rilixai.Spec` binds four things the optimizer needs:
the seed :class:`~rilixai.OptimizationTargets` (``system_prompt`` +
``task_template`` + ``resum_summary_prompt``), a
:class:`~apex_agents.data.dataset.ApexAgentsDataLoader` that turns
uploaded JSONL rows into cases, the async ``run_case`` that drives the
ReAct agent + rubric judge, and the :class:`ApexAgentsScorer`.

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

from collections.abc import Callable
from typing import Any

from rilixai import CaseDataLoader, OptimizationContext, OptimizationTargets, Spec, spec

from ..agent.prompts import apex_agents_seed_targets
from ..config import ApexAgentsConfig
from ..data.dataset import ApexAgentsDataLoader, ApexAgentsRecord
from .metrics import ApexAgentsScorer
from .runtime import build_apex_agents_run_case


def build_apex_agents_spec(
    *,
    seed_targets: OptimizationTargets | None = None,
    config: ApexAgentsConfig | None = None,
    agent: Any | None = None,
    world_factory: Callable[[Any], Any] | None = None,
    model_factory: Callable[[str, float], Any] | None = None,
    judge: Callable[[str, str, str], bool] | None = None,
    name: str = "apex-agents",
    field_weights: dict[str, float] | None = None,
    data_loader: CaseDataLoader[ApexAgentsRecord] | None = None,
) -> Spec:
    """Build a ready-to-run rilixai :class:`~rilixai.Spec` for APEX-Agents.

    ``world_factory`` is required at run time unless a pre-built
    ``agent`` is supplied — production code passes the HF world
    builder; tests pass a closure yielding :class:`FakeWorld`.
    ``judge`` defaults to the litellm rubric judge; tests inject a
    stub.

    The optimizer sources cases from ``data_loader`` (uploaded JSONL);
    ``ApexAgentsDataLoader`` maps each row to one :class:`~rilixai.Case`
    whose ``run_metrics.trace_evidence.per_component_feedback`` carries
    the reflection signal scalar field scores cannot represent.
    """
    cfg = config or ApexAgentsConfig()
    seed = seed_targets or apex_agents_seed_targets()
    run_case = build_apex_agents_run_case(
        config=cfg,
        agent=agent,
        world_factory=world_factory,
        model_factory=model_factory,
        judge=judge,
    )
    return Spec(
        name=name,
        seed_targets=seed,
        data_loader=data_loader or ApexAgentsDataLoader(),
        run_case=run_case,
        scorer=ApexAgentsScorer(field_weights=field_weights),
    )


# ─── rilixai Modal sandbox entry point ─────────────────────────────────


# Defaults the sandbox build_spec applies when ctx.config omits a key.
# See README for the full key reference (GEPA vs cookbook split).
_DEFAULT_SANDBOX_CONFIG: dict[str, Any] = {
    "task_model": "openai/gpt-4.1-mini-2025-04-14",
    "task_temperature": 0.0,
    "judge_model": "gemini/gemini-2.5-flash",
    "max_steps": 60,
    "cost_limit": 3.0,
}


@spec(
    name="apex-agents",
    description="APEX-Agents (Mercor law/IB) — faithful ReAct toolbelt agent + GEPA",
    metadata={"benchmark": "apex_agents", "agent_kind": "react_toolbelt"},
    dataset_schema=ApexAgentsDataLoader.dataset_schema,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Spec factory for the rilixai sandbox path. See README for usage.

    No ``version=...`` argument here: ``sandbox.py --build`` supplies
    the push-time version (defaulting to ``v<short_sha>``) via
    ``rilixai push --version`` and then promotes the freshly-pushed
    row to ``apex-agents@production``. Trigger calls reference
    ``apex-agents@production`` so rilixai resolves the current
    promoted version server-side — no rilix-side / cookbook-side
    version bumps for routine deploys.

    Under the SDK-only shape the optimizer sources cases from the
    uploaded JSONL dataset via :class:`ApexAgentsDataLoader`. The
    recipe's ReAct-agent knobs travel under ``ctx.config["extra"]`` (the
    launch contract reserves the top level for optimizer-owned
    settings), merged over :data:`_DEFAULT_SANDBOX_CONFIG`. When the
    optimizer selects a task model it arrives as ``ctx.model`` and
    overrides the configured ``task_model``.
    """
    extra = dict(ctx.config.get("extra") or {}) if ctx.config else {}
    cfg_in: dict[str, Any] = {**_DEFAULT_SANDBOX_CONFIG, **extra}
    apex_cfg = ApexAgentsConfig(
        task_model=str(ctx.model) if ctx.model else str(cfg_in["task_model"]),
        task_temperature=float(cfg_in["task_temperature"]),
        judge_model=str(cfg_in["judge_model"]),
        max_steps=int(cfg_in["max_steps"]),
        cost_limit=float(cfg_in["cost_limit"]),
    )

    # World factory: lazy HF download per case. Network is available
    # inside the sandbox container; ``build_world_factory`` extracts the
    # per-case world zip + per-task input files on demand.
    from ..agent.world.world import build_world_factory

    return build_apex_agents_spec(
        config=apex_cfg,
        world_factory=build_world_factory(),
    )


__all__ = ["build_apex_agents_spec", "build_spec"]
