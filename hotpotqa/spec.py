"""Factory that assembles a :class:`PromptOptimizationSpec` for HotpotQA.

The spec is the recommended public integration point with rilixai's optimizer:
once built, the same ``run_optimization_from_spec`` and
``build_adapter_from_spec`` helpers used for production extraction tasks
run HotpotQA end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rilixai.prompt_optimization.models import Case, PromptCandidate
from rilixai.prompt_optimization.protocols import EvaluationProfile
from rilixai.prompt_optimization.spec import PromptOptimizationSpec

from .agent.prompts import hotpotqa_pydantic_agent_seed_candidate
from .metrics import (
    HOTPOTQA_FIELD_WEIGHTS,
    HotpotQAMetricsCalculator,
    build_hotpotqa_field_extractor,
)
from .pipeline import HotpotQAPipelineConfig, build_hotpotqa_runtime


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
    pipeline_config: HotpotQAPipelineConfig | None = None,
    pydantic_agent: Any | None = None,
    name: str = "hotpotqa",
    user_id: str = "__hotpotqa_benchmark__",
    model: str | None = None,
    max_concurrency: int = 8,
    reflection_evidence_mode: str = "curated_plus_trace",
    field_weights: dict[str, float] | None = None,
) -> PromptOptimizationSpec:
    """Build a ready-to-run :class:`PromptOptimizationSpec` for HotpotQA.

    Pass ``pipeline_config.pydantic_agent_model`` (a PydanticAI model
    spec like ``"openai:gpt-4.1-mini"``) or a pre-built ``pydantic_agent``
    instance. ``cases_by_split`` must include at least ``train`` and
    (``validation`` or ``val``) keys for optimization; evaluation-only
    uses can supply just one split. ``curated_plus_trace`` is the
    default reflection mode because the runtime populates per-tool
    ``trace_evidence`` that scalar field scores cannot represent.
    """
    cfg = pipeline_config or HotpotQAPipelineConfig()
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
