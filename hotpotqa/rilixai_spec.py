"""The whole HotpotQA → rilixai integration: one ``@spec`` runner class.

``rilixai push`` targets this file. :class:`HotpotQARunner` is the entire
sandbox integration — rilixai assembles the metrics calculator, seed
candidate, and per-component feedback from the declarations on ``@spec`` and
the runner's ``result_context`` hook (which emits the paper-style trace). The
``@spec`` decorator builds the :class:`PromptOptimizationSpec` from the runner
class; rilixai resolves it via ``load_spec_from_target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from rilixai import spec
from rilixai.adapters import BaseCaseRunner
from rilixai.metrics import BaseMetricsCalculator, FieldConfig
from rilixai.prompt_optimization.models import Case

from .agent.prompts import DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT, DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT
from .agent.types import HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import HotpotQARecord, load_hotpotqa_paper_split
from .data.eval import f1_score
from .feedback import HotpotQAFeedback
from .metrics import build_agent_run_metrics


@dataclass
class _ReportableHotpotQAOutput(HotpotQAAgentOutput):
    """Hotpot output with a compact JSON-safe view for rilixai artifacts."""

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return _prediction_for_rilixai(self)


# ─── Scoring ────────────────────────────────────────────────────────────


def _hotpot_f1(predicted: Any, expected: Any) -> float:
    return float(f1_score(predicted, expected))


class HotpotQAMetrics(BaseMetricsCalculator):
    """Scoring for HotpotQA, reusing the paper-faithful answer scorers.

    Exact-match drives candidate selection (weight 1.0); token F1 and
    supporting-title recall are computed as diagnostics (weight 0.0). EM and F1
    are declared as separate fields rather than a single multi-comparator field
    so they can carry the paper's distinct weights.
    """

    # FieldConfig can reference rilixai's shipped comparators directly:
    # exact_match, f1_score, set_recall, set_f1, numeric_close, llm_judge.
    # Register custom comparator names in ``comparators`` below when a domain
    # scorer differs from the shipped catalog. HotpotQA uses shipped
    # ``exact_match``/``set_recall`` and a custom ``hotpot_f1`` because the
    # official benchmark F1 handles yes/no/noanswer mismatches specially.
    fields = [
        FieldConfig(name="answer", comparators="exact_match", weight=1.0),
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


def _sandbox_config(ctx: Any) -> HotpotQASandboxConfig:
    raw = getattr(ctx, "config", None)
    if isinstance(raw, HotpotQASandboxConfig):
        return raw
    return HotpotQASandboxConfig.model_validate({} if raw is None else raw)


# ─── The integration: one @spec runner class ────────────────────────────


@spec(
    name="hotpotqa-agent",
    description="HotpotQA multi-hop QA — PydanticAI tool-using agent + GEPA",
    metadata={"benchmark": "hotpotqa", "agent_kind": "pydantic_ai"},
    task_type="hotpotqa_pydantic_agent",
    config_schema=HotpotQASandboxConfig,
    field_configs=HotpotQAMetrics,
    feedback=HotpotQAFeedback,
    # No explicit seed: rilixai auto-reads it from the prompts declared in
    # BaseCaseRunner.__init__ below.
    # reflection_evidence_mode is kept (rilixai's default is "curated");
    # this agent emits rich trace_evidence the reflection LM should use.
    reflection_evidence_mode="curated_plus_trace",
)
class HotpotQARunner(BaseCaseRunner[HotpotQARecord, HotpotQAAgentOutput]):
    """The entire HotpotQA integration: one runner the rilixai sandbox drives.

    No ``version=`` on ``@spec`` — hosted pushes pass ``--version`` (CI uses
    ``v<short_sha>``) and promote that immutable build to
    ``hotpotqa-agent@production``, which trigger calls reference.
    """

    def __init__(self, ctx: Any) -> None:
        sandbox_cfg = _sandbox_config(ctx)
        self._sandbox_cfg = sandbox_cfg
        self.cfg = HotpotQAConfig(
            retrieval_mode=sandbox_cfg.retrieval_mode,
            retrieve_k=sandbox_cfg.retrieve_k,
            max_iters=sandbox_cfg.max_iters,
            pydantic_agent_model=sandbox_cfg.pydantic_agent_model,
            pydantic_agent_temperature=sandbox_cfg.task_temperature,
        )
        super().__init__(
            prompts={
                "policy_prompt": DEFAULT_PYDANTIC_AGENT_POLICY_PROMPT,
                "summarize_prompt": DEFAULT_PYDANTIC_AGENT_SUMMARIZE_PROMPT,
            }
        )

    async def run_case(self, record: HotpotQARecord) -> HotpotQAAgentOutput:
        from .agent.agent import HotpotQAPydanticAgent
        from .agent.retrieval import build_retrieve_k_fn_for_case

        retrieve_k_fn = build_retrieve_k_fn_for_case(record=record, cfg=self.cfg)
        sandbox_cfg = self._sandbox_cfg
        agent = HotpotQAPydanticAgent(
            model=sandbox_cfg.pydantic_agent_model,
            summarize_model=_bare_openai_model(sandbox_cfg.pydantic_agent_model),
            top_k=sandbox_cfg.retrieve_k,
            max_iters=sandbox_cfg.max_iters,
            temperature=sandbox_cfg.task_temperature,
            policy_prompt=self.prompt("policy_prompt"),
            summarize_prompt=self.prompt("summarize_prompt"),
        )
        return await agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
            retrieve_k_fn=retrieve_k_fn,
        )

    def result_context(
        self,
        record: HotpotQARecord,
        output: HotpotQAAgentOutput,
    ) -> dict[str, Any]:
        # Domain-specific result context (per-hop retrieval reasoning,
        # documents-remaining, missing/spurious titles) is richer than the base
        # hook set. Per-component feedback flows automatically from
        # ``@spec(feedback=HotpotQAFeedback)`` into trace_evidence.
        return build_agent_run_metrics(
            record=record,
            output=output,
            agent_kind="pydantic",
            config=self.cfg,
        )

    def result_output(self, output: HotpotQAAgentOutput) -> HotpotQAAgentOutput:
        return _reportable_output(output)

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


def _prediction_for_rilixai(output: HotpotQAAgentOutput) -> dict[str, Any]:
    """Expose only JSON-safe prediction fields rilixai scores and reports."""
    return {
        "answer": output.answer,
        "retrieved_titles": output.retrieved_titles,
    }


def _reportable_output(output: HotpotQAAgentOutput) -> HotpotQAAgentOutput:
    return _ReportableHotpotQAOutput(
        answer=output.answer,
        retrieved_paragraphs=output.retrieved_paragraphs,
        tool_calls=output.tool_calls,
    )
