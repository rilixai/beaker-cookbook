"""Factory that assembles a rilixai :class:`~rilixai.Spec` for HotpotQA.

The SDK :class:`~rilixai.Spec` binds four things the optimizer needs: the
seed :class:`~rilixai.OptimizationTargets` (``policy_prompt`` +
``summarize_prompt``), a :class:`~hotpotqa.data.dataset.HotpotQADataLoader`
that turns uploaded JSONL rows into cases, the async ``run_case`` that drives
the PydanticAI agent, and the :class:`HotpotQAScorer`.

This module also exports the rilixai Modal-sandbox entry point
:func:`build_spec` — a ``@spec(name="hotpotqa-agent")``-decorated factory
that rilixai's sandbox dispatcher invokes once per remote optimization run.
See the cookbook README for the push + promote + trigger flow.
"""

from __future__ import annotations

from typing import Any

from rilixai import CaseDataLoader, OptimizationContext, OptimizationTargets, Spec, spec

from ..agent.prompts import hotpotqa_pydantic_agent_seed_targets
from ..config import HotpotQAConfig
from ..data.dataset import HotpotQADataLoader, HotpotQARecord
from .metrics import HotpotQAScorer
from .runtime import build_hotpotqa_run_case


def build_hotpotqa_spec(
    *,
    seed_targets: OptimizationTargets | None = None,
    config: HotpotQAConfig | None = None,
    pydantic_agent: Any | None = None,
    name: str = "hotpotqa",
    field_weights: dict[str, float] | None = None,
    data_loader: CaseDataLoader[HotpotQARecord] | None = None,
) -> Spec:
    """Build a ready-to-run rilixai :class:`~rilixai.Spec` for HotpotQA.

    Pass ``config.pydantic_agent_model`` (a PydanticAI model spec like
    ``"openai:gpt-4.1-mini"``) or a pre-built ``pydantic_agent``
    instance. The optimizer sources cases from ``data_loader`` (uploaded
    JSONL); :class:`HotpotQADataLoader` maps each row to one
    :class:`~rilixai.Case` whose ``run_metrics.trace_evidence`` carries
    the per-tool reflection signal scalar field scores cannot represent.
    """
    cfg = config or HotpotQAConfig()
    seed = seed_targets or hotpotqa_pydantic_agent_seed_targets()
    run_case = build_hotpotqa_run_case(
        config=cfg,
        pydantic_agent=pydantic_agent,
    )
    return Spec(
        name=name,
        seed_targets=seed,
        data_loader=data_loader or HotpotQADataLoader(),
        run_case=run_case,
        scorer=HotpotQAScorer(field_weights=field_weights),
    )


# ─── rilixai Modal sandbox entry point ─────────────────────────────────


# Defaults the sandbox build_spec applies when ctx.config omits a key.
# See README for the full key reference (GEPA vs cookbook split).
_DEFAULT_SANDBOX_CONFIG: dict[str, Any] = {
    "retrieval_mode": "distractor",
    "retrieve_k": 7,
    "max_iters": 8,
    "pydantic_agent_model": "openai:gpt-4.1-mini",
    "task_temperature": 0.0,
    "max_concurrency": 4,
}


@spec(
    name="hotpotqa-agent",
    description="HotpotQA multi-hop QA — PydanticAI tool-using agent + GEPA",
    metadata={"benchmark": "hotpotqa", "agent_kind": "pydantic_ai"},
    dataset_schema=HotpotQADataLoader.dataset_schema,
)
def build_spec(ctx: OptimizationContext) -> Spec:
    """Spec factory for the rilixai Modal sandbox path. See README for usage.

    Intentionally no ``version=...`` argument. ``sandbox.py --build``
    supplies the push-time version (defaulting to ``v<short_sha>``)
    via ``rilixai push --version`` and then promotes the freshly-
    pushed row to ``hotpotqa-agent@production``. Trigger calls
    reference ``hotpotqa-agent@production`` so rilixai resolves the
    current promoted version server-side — no rilix-side / cookbook-
    side version bumps for routine deploys.

    Under the SDK-only shape the optimizer sources cases from the
    uploaded JSONL dataset via :class:`HotpotQADataLoader`; the agent
    knobs still come from ``ctx.config`` (merged over
    :data:`_DEFAULT_SANDBOX_CONFIG`).
    """
    cfg_in: dict[str, Any] = {**_DEFAULT_SANDBOX_CONFIG, **dict(ctx.config or {})}
    hotpot_cfg = HotpotQAConfig(
        retrieval_mode=cfg_in["retrieval_mode"],
        retrieve_k=int(cfg_in["retrieve_k"]),
        max_iters=int(cfg_in["max_iters"]),
        pydantic_agent_model=str(cfg_in["pydantic_agent_model"]),
        pydantic_agent_temperature=float(cfg_in["task_temperature"]),
    )
    return build_hotpotqa_spec(config=hotpot_cfg)


__all__ = ["build_hotpotqa_spec", "build_spec"]
