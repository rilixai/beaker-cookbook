"""HotpotQA spec wiring for rilixai's optimizer.

Two entry points live here:

* :func:`build_hotpotqa_spec` — the explicit factory the local CLI uses to
  assemble a :class:`PromptOptimizationSpec` by hand (full control over splits,
  field weights, and a pre-built agent for tests).
* :class:`HotpotQARunner` — the class-style ``@spec`` form the rilixai Modal
  sandbox runs. The whole sandbox integration is this one
  :class:`~rilixai.adapters.BaseSampleRunner` subclass: rilixai assembles the
  metrics calculator, seed candidate, and feedback from the declarations on
  the decorator + the methods on the class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel
from rilixai import spec
from rilixai.adapters import BaseSampleRunner, CallableApplier, per_component_feedback
from rilixai.metrics import BaseMetricsCalculator, FieldConfig
from rilixai.prompt_optimization.models import PromptCandidate, Sample
from rilixai.prompt_optimization.protocols import EvaluationProfile
from rilixai.prompt_optimization.spec import PromptOptimizationSpec

from ..agent.prompts import hotpotqa_pydantic_agent_seed_candidate
from ..agent.types import HotpotQAAgentOutput
from ..config import HotpotQAConfig
from ..data.dataset import HotpotQARecord, load_hotpotqa_paper_split
from ..data.eval import exact_match_score, f1_score
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
    samples_by_split: dict[str, Sequence[Sample]],
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
    instance. ``samples_by_split`` must include at least ``train`` and
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
        samples_by_split=samples_by_split,
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


# ─── rilixai Modal sandbox entry point — class-style @spec ──────────────


class HotpotQASandboxConfig(BaseModel):
    """Typed sandbox run config. rilixai validates ``ctx.config`` against this.

    A typo in a trigger's config key fails fast with a Pydantic error rather
    than silently falling back to a default. ``distractor`` retrieval is the
    default so cold starts skip the 5GB fullwiki download; pass
    ``retrieval_mode="fullwiki"`` for paper parity.
    """

    retrieval_mode: Literal["distractor", "fullwiki"] = "distractor"
    retrieve_k: int = 7
    max_iters: int = 8
    train_size: int = 50
    val_size: int = 100
    pydantic_agent_model: str = "openai:gpt-4.1-mini"
    task_temperature: float = 0.0
    max_concurrency: int = 4


def _hotpot_exact_match(predicted: Any, expected: Any) -> float:
    return 1.0 if exact_match_score(predicted, expected) else 0.0


def _hotpot_f1(predicted: Any, expected: Any) -> float:
    return float(f1_score(predicted, expected))


class HotpotQASandboxMetrics(BaseMetricsCalculator):
    """Scoring for the sandbox runner, reusing the paper-faithful answer scorers.

    ``answer`` emits two scores (paper EM + token F1) from the canonical
    HotpotQA evaluator; ``titles_recall`` measures multi-hop retrieval coverage.
    """

    fields = [
        FieldConfig(name="answer", comparators=["hotpot_exact_match", "hotpot_f1"]),
        FieldConfig(
            name="titles_recall",
            extract_from="retrieved_titles",
            compare_to="supporting_titles",
            comparators="set_recall",
        ),
    ]
    comparators = {
        "hotpot_exact_match": _hotpot_exact_match,
        "hotpot_f1": _hotpot_f1,
    }


class HotpotQASandboxFeedback:
    """Per-component narratives for the reflection LM, reusing the agent feedback.

    Delegates to :func:`build_agent_per_component_feedback`, which builds the
    paper-style policy / summarize narratives from the agent's tool-call trace.
    """

    def __init__(self) -> None:
        # Imported lazily so the feedback module's transitive deps don't load
        # at spec-import time.
        from .feedback import build_agent_per_component_feedback

        self._build = build_agent_per_component_feedback

    @per_component_feedback("policy_prompt")
    def _policy(self, sample: Any, output: HotpotQAAgentOutput) -> str:
        return self._build(record=sample.input, output=output).get("policy_prompt", "")

    @per_component_feedback("summarize_prompt")
    def _summarize(self, sample: Any, output: HotpotQAAgentOutput) -> str:
        return self._build(record=sample.input, output=output).get("summarize_prompt", "")


@spec(
    name="hotpotqa-agent",
    description="HotpotQA multi-hop QA — PydanticAI tool-using agent + GEPA",
    metadata={"benchmark": "hotpotqa", "agent_kind": "pydantic_ai"},
    task_type="hotpotqa_pydantic_agent",
    config_schema=HotpotQASandboxConfig,
    field_configs=HotpotQASandboxMetrics,
    feedback=HotpotQASandboxFeedback,
    seed=hotpotqa_pydantic_agent_seed_candidate(),
    reflection_evidence_mode="curated_plus_trace",
    max_concurrency=4,
)
class HotpotQARunner(BaseSampleRunner[HotpotQARecord, HotpotQAAgentOutput]):
    """The entire sandbox integration: one runner the rilixai sandbox drives.

    No ``version=`` on ``@spec`` — ``rilixai push`` supplies the push-time
    version (defaulting to ``v<short_sha>``) and promotes it to
    ``hotpotqa-agent@production``, which trigger calls reference.
    """

    def __init__(self, ctx: Any) -> None:
        cfg = ctx.config if isinstance(ctx.config, HotpotQASandboxConfig) else HotpotQASandboxConfig()
        self._sandbox_cfg = cfg
        self.cfg = HotpotQAConfig(
            retrieval_mode=cfg.retrieval_mode,
            retrieve_k=cfg.retrieve_k,
            max_iters=cfg.max_iters,
            pydantic_agent_model=cfg.pydantic_agent_model,
            pydantic_agent_temperature=cfg.task_temperature,
        )
        # Defer the pydantic-ai-touching import until the runner is built.
        from ..agent.agent import HotpotQAPydanticAgent

        self.agent = HotpotQAPydanticAgent(
            model=cfg.pydantic_agent_model,
            summarize_model=_bare_openai_model(cfg.pydantic_agent_model),
            top_k=cfg.retrieve_k,
            max_iters=cfg.max_iters,
            temperature=cfg.task_temperature,
        )
        super().__init__(
            applier=CallableApplier(
                apply=self.agent.apply_candidate,
                read=lambda: dict(hotpotqa_pydantic_agent_seed_candidate().components),
            )
        )

    async def run_sample(self, record: HotpotQARecord) -> HotpotQAAgentOutput:
        from ..agent.retrieval import build_retrieve_k_fn_for_case

        retrieve_k_fn = build_retrieve_k_fn_for_case(record=record, cfg=self.cfg)
        return await self.agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
            retrieve_k_fn=retrieve_k_fn,
        )

    def _collect_tool_calls(self, output: HotpotQAAgentOutput) -> list[dict[str, Any]]:
        return [{"tool": tc.tool_name, "args": dict(tc.tool_args)} for tc in output.tool_calls]

    def samples_by_split(self, ctx: Any) -> dict[str, list[Sample]]:
        hf_config = "fullwiki" if self._sandbox_cfg.retrieval_mode == "fullwiki" else "distractor"
        return {
            "train": list(
                load_hotpotqa_paper_split("train", max_cases=self._sandbox_cfg.train_size, config=hf_config)
            ),
            "validation": list(
                load_hotpotqa_paper_split("validation", max_cases=self._sandbox_cfg.val_size, config=hf_config)
            ),
        }


def _bare_openai_model(pydantic_spec: str) -> str:
    """Strip the provider prefix from a PydanticAI model spec.

    PydanticAI uses ``"openai:gpt-4.1-mini"``; the raw OpenAI summarize call
    wants ``"gpt-4.1-mini"``. Returns the original string when no ``:`` is
    present so already-bare specs pass through unchanged.
    """
    _, separator, model = pydantic_spec.partition(":")
    return model if separator else pydantic_spec
