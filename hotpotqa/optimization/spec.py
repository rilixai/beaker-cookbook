"""Factory that assembles a :class:`PromptOptimizationSpec` for HotpotQA.

The spec is the recommended public integration point with rilixai's optimizer:
once built, the same ``run_optimization_from_spec`` and
``build_adapter_from_spec`` helpers used for production extraction tasks
run HotpotQA end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rilixai import spec
from rilixai.prompt_optimization.models import Case, PromptCandidate
from rilixai.prompt_optimization.protocols import EvaluationProfile
from rilixai.prompt_optimization.spec import OptimizationContext, PromptOptimizationSpec

from ..agent.prompts import hotpotqa_pydantic_agent_seed_candidate
from ..config import HotpotQAConfig
from ..data.dataset import load_hotpotqa_paper_split
from .metrics import (
    HOTPOTQA_FIELD_WEIGHTS,
    HotpotQAMetricsCalculator,
    build_hotpotqa_field_extractor,
)
from .runtime import build_hotpotqa_runtime


_HOTPOTQA_PROFILE_KEY = "hotpotqa"


def _hotpotqa_agent_resolver(**_: Any) -> tuple[None, None]:
    """The runtime owns the agent + LM, so adapter agent resolution is a no-op."""
    return (None, None)


def _build_hotpotqa_profile_resolver(
    metrics: HotpotQAMetricsCalculator,
    field_weights: dict[str, float] | None = None,
) -> Any:
    profile = EvaluationProfile(
        profile_key=_HOTPOTQA_PROFILE_KEY,
        metrics_calculator=metrics,
        field_weights=dict(field_weights or HOTPOTQA_FIELD_WEIGHTS),
    )

    def _resolver(**_: Any) -> EvaluationProfile:
        return profile

    return _resolver


def build_hotpotqa_spec(
    *,
    cases_by_split: dict[str, Sequence[Case]],
    seed_candidate: PromptCandidate | None = None,
    config: HotpotQAConfig | None = None,
    pydantic_agent: Any | None = None,
    name: str = "hotpotqa",
    user_id: str = "__hotpotqa_benchmark__",
    model: str | None = None,
    max_concurrency: int = 8,
    reflection_evidence_mode: str = "curated_plus_trace",
    field_weights: dict[str, float] | None = None,
) -> PromptOptimizationSpec:
    """Build a ready-to-run :class:`PromptOptimizationSpec` for HotpotQA.

    Pass ``config.pydantic_agent_model`` (a PydanticAI model spec like
    ``"openai:gpt-4.1-mini"``) or a pre-built ``pydantic_agent``
    instance. ``cases_by_split`` must include at least ``train`` and
    (``validation`` or ``val``) keys for optimization; evaluation-only
    uses can supply just one split. ``curated_plus_trace`` is the
    default reflection mode because the runtime populates per-tool
    ``trace_evidence`` that scalar field scores cannot represent.
    """
    cfg = config or HotpotQAConfig()
    seed = seed_candidate or hotpotqa_pydantic_agent_seed_candidate()
    metrics = HotpotQAMetricsCalculator()
    runtime = build_hotpotqa_runtime(
        config=cfg,
        pydantic_agent=pydantic_agent,
    )
    return PromptOptimizationSpec(
        cases_by_split=cases_by_split,
        seed_candidate=seed,
        extraction_runtime=runtime,
        agent_resolver=_hotpotqa_agent_resolver,
        field_extractor=build_hotpotqa_field_extractor(),
        evaluation_profile_resolver=_build_hotpotqa_profile_resolver(metrics, field_weights),
        name=name,
        user_id=user_id,
        model=model,
        task_type="hotpotqa_pydantic_agent",
        max_concurrency=max_concurrency,
        reflection_evidence_mode=reflection_evidence_mode,
    )


# ─── rilixai Modal sandbox entry point ─────────────────────────────────


# Defaults the sandbox build_spec applies when ctx.config omits a key.
# See README for the full key reference (GEPA vs cookbook split).
_DEFAULT_SANDBOX_CONFIG: dict[str, Any] = {
    "retrieval_mode": "distractor",
    "retrieve_k": 7,
    "max_iters": 8,
    "train_size": 50,
    "val_size": 100,
    "pydantic_agent_model": "openai:gpt-4.1-mini",
    "task_temperature": 0.0,
    "max_concurrency": 4,
}


@spec(
    name="hotpotqa-agent",
    description="HotpotQA multi-hop QA — PydanticAI tool-using agent + GEPA",
    metadata={"benchmark": "hotpotqa", "agent_kind": "pydantic_ai"},
)
def build_spec(ctx: OptimizationContext) -> PromptOptimizationSpec:
    """Spec factory for the rilixai Modal sandbox path. See README for usage.

    Intentionally no ``version=...`` argument. ``sandbox.py --build``
    supplies the push-time version (defaulting to ``v<short_sha>``)
    via ``rilixai push --version`` and then promotes the freshly-
    pushed row to ``hotpotqa-agent@production``. Trigger calls
    reference ``hotpotqa-agent@production`` so rilixai resolves the
    current promoted version server-side — no rilix-side / cookbook-
    side version bumps for routine deploys.
    """
    cfg_in = {**_DEFAULT_SANDBOX_CONFIG, **dict(ctx.config or {})}
    hotpot_cfg = HotpotQAConfig(
        retrieval_mode=cfg_in["retrieval_mode"],
        retrieve_k=int(cfg_in["retrieve_k"]),
        max_iters=int(cfg_in["max_iters"]),
        pydantic_agent_model=str(cfg_in["pydantic_agent_model"]),
        pydantic_agent_temperature=float(cfg_in["task_temperature"]),
    )
    hf_config = "fullwiki" if hotpot_cfg.retrieval_mode == "fullwiki" else "distractor"
    splits = {
        "train": load_hotpotqa_paper_split(
            "train",
            max_cases=int(cfg_in["train_size"]),
            config=hf_config,
        ),
        "validation": load_hotpotqa_paper_split(
            "validation",
            max_cases=int(cfg_in["val_size"]),
            config=hf_config,
        ),
    }
    return build_hotpotqa_spec(
        cases_by_split={k: list(v) for k, v in splits.items()},
        config=hotpot_cfg,
        model=cfg_in.get("model") or ctx.model,
        max_concurrency=int(cfg_in["max_concurrency"]),
    )
