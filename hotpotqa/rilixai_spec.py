"""The whole HotpotQA → rilixai integration: one ``@spec`` runner class.

``rilixai push`` targets this file. :class:`HotpotQARunner` is the entire
sandbox integration — rilixai assembles the metrics calculator, seed
candidate, and per-component feedback from the declarations on ``@spec`` and
the runner's own ``_package_result`` (which emits the paper-style trace). The
``@spec`` decorator builds the :class:`PromptOptimizationSpec` from the runner
class; rilixai resolves it via ``load_spec_from_target``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from rilixai import spec
from rilixai.adapters import AttributeApplier, BaseCaseRunner, CaseRunResult
from rilixai.metrics import BaseMetricsCalculator, FieldConfig
from rilixai.prompt_optimization.models import Case

from .agent.prompts import DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT, DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT
from .agent.types import HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import HotpotQARecord, load_hotpotqa_paper_split
from .data.eval import exact_match_score, f1_score
from .feedback import HotpotQAFeedback
from .metrics import build_agent_run_metrics


# ─── Optimizable prompt components ──────────────────────────────────────


POLICY_PROMPT_COMPONENT = "policy_prompt"
SUMMARIZE_PROMPT_COMPONENT = "summarize_prompt"


@dataclass
class _HotpotQAPrompts:
    """Runner-owned prompt state that rilixai candidates mutate."""

    policy_prompt: str = DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT
    summarize_prompt: str = DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT


# ─── Scoring ────────────────────────────────────────────────────────────


def _hotpot_exact_match(predicted: Any, expected: Any) -> float:
    return 1.0 if exact_match_score(predicted, expected) else 0.0


def _hotpot_f1(predicted: Any, expected: Any) -> float:
    return float(f1_score(predicted, expected))


class HotpotQAMetrics(BaseMetricsCalculator):
    """Scoring for HotpotQA, reusing the paper-faithful answer scorers.

    Exact-match drives candidate selection (weight 1.0); token F1 and
    supporting-title recall are computed as diagnostics (weight 0.0). EM and F1
    are declared as separate fields rather than a single multi-comparator field
    so they can carry the paper's distinct weights.
    """

    fields = [
        FieldConfig(name="answer", comparators="hotpot_exact_match", weight=1.0),
        FieldConfig(name="answer_f1", extract_from="answer", comparators="hotpot_f1", weight=0.0),
        FieldConfig(
            name="titles_recall",
            extract_from="retrieved_titles",
            compare_to="supporting_titles",
            comparators="set_recall",
            weight=0.0,
        ),
    ]
    comparators = {
        "hotpot_exact_match": _hotpot_exact_match,
        "hotpot_f1": _hotpot_f1,
    }


# ─── Sandbox run config ─────────────────────────────────────────────────


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


# ─── The integration: one @spec runner class ────────────────────────────


@spec(
    name="hotpotqa-agent",
    description="HotpotQA multi-hop QA — PydanticAI tool-using agent + GEPA",
    metadata={"benchmark": "hotpotqa", "agent_kind": "pydantic_ai"},
    task_type="hotpotqa_pydantic_agent",
    config_schema=HotpotQASandboxConfig,
    field_configs=HotpotQAMetrics,
    feedback=HotpotQAFeedback,
    # No explicit seed: rilixai auto-reads it from runner-owned prompt state
    # via the applier's read() at spec-build time.
    # reflection_evidence_mode is kept (rilixai's default is "curated");
    # this agent emits rich trace_evidence the reflection LM should use.
    reflection_evidence_mode="curated_plus_trace",
    # rilixai's default max_concurrency is 8; 4 is the cost-bounded demo value.
    max_concurrency=4,
)
class HotpotQARunner(BaseCaseRunner[HotpotQARecord, HotpotQAAgentOutput]):
    """The entire HotpotQA integration: one runner the rilixai sandbox drives.

    No ``version=`` on ``@spec`` — ``rilixai push`` supplies the push-time
    version (defaulting to ``v<short_sha>``) and promotes it to
    ``hotpotqa-agent@production``, which trigger calls reference.
    """

    def __init__(self, ctx: Any) -> None:
        sandbox_cfg = ctx.config if isinstance(ctx.config, HotpotQASandboxConfig) else HotpotQASandboxConfig()
        self._sandbox_cfg = sandbox_cfg
        self.cfg = HotpotQAConfig(
            retrieval_mode=sandbox_cfg.retrieval_mode,
            retrieve_k=sandbox_cfg.retrieve_k,
            max_iters=sandbox_cfg.max_iters,
            pydantic_agent_model=sandbox_cfg.pydantic_agent_model,
            pydantic_agent_temperature=sandbox_cfg.task_temperature,
        )
        self.prompts = _HotpotQAPrompts()
        super().__init__(
            applier=AttributeApplier(
                target=self.prompts,
                mapping={
                    POLICY_PROMPT_COMPONENT: "policy_prompt",
                    SUMMARIZE_PROMPT_COMPONENT: "summarize_prompt",
                },
            )
        )

    async def run_case(self, record: HotpotQARecord) -> HotpotQAAgentOutput:
        from .agent.retrieval import build_retrieve_k_fn_for_case

        retrieve_k_fn = build_retrieve_k_fn_for_case(record=record, cfg=self.cfg)
        agent = self._build_agent()
        return await agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
            retrieve_k_fn=retrieve_k_fn,
        )

    def _build_agent(self) -> Any:
        # Defer the pydantic-ai-touching import until a case actually runs.
        from .agent.agent import HotpotQAPydanticAgent

        return HotpotQAPydanticAgent(
            model=self._sandbox_cfg.pydantic_agent_model,
            summarize_model=_bare_openai_model(self._sandbox_cfg.pydantic_agent_model),
            top_k=self._sandbox_cfg.retrieve_k,
            max_iters=self._sandbox_cfg.max_iters,
            temperature=self._sandbox_cfg.task_temperature,
            policy_prompt=self.prompts.policy_prompt,
            summarize_prompt=self.prompts.summarize_prompt,
        )

    def _package_result(
        self,
        record: HotpotQARecord,
        output: HotpotQAAgentOutput,
        runtime_kwargs: Mapping[str, Any],
    ) -> CaseRunResult[HotpotQAAgentOutput]:
        # Domain-specific trace evidence (per-hop retrieval reasoning,
        # documents-remaining, missing/spurious titles) is richer than the base
        # hook set, so build it here. Per-component feedback flows through
        # ``@spec(feedback=HotpotQAFeedback)`` — merge it into the trace.
        run_metrics = build_agent_run_metrics(
            record=record,
            output=output,
            agent_kind="pydantic",
            config=self.cfg,
        )
        feedback = self._build_feedback(record, output, runtime_kwargs)
        if feedback:
            run_metrics.setdefault("trace_evidence", {})["per_component_feedback"] = feedback
        return CaseRunResult(output=output, run_metrics=run_metrics)

    def cases_by_split(self, ctx: Any) -> dict[str, list[Case]]:
        hf_config = "fullwiki" if self._sandbox_cfg.retrieval_mode == "fullwiki" else "distractor"
        return {
            "train": list(
                load_hotpotqa_paper_split("train", max_cases=self._sandbox_cfg.train_size, config=hf_config)
            ),
            "validation": list(
                load_hotpotqa_paper_split("validation", max_cases=self._sandbox_cfg.val_size, config=hf_config)
            ),
        }


# ─── Helpers ────────────────────────────────────────────────────────────


def _bare_openai_model(pydantic_spec: str) -> str:
    """Strip the provider prefix from a PydanticAI model spec.

    PydanticAI uses ``"openai:gpt-4.1-mini"``; the raw OpenAI summarize call
    wants ``"gpt-4.1-mini"``. Returns the original string when no ``:`` is
    present so already-bare specs pass through unchanged.
    """
    _, separator, model = pydantic_spec.partition(":")
    return model if separator else pydantic_spec
