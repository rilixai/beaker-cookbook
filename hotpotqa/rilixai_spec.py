"""The whole HotpotQA → rilixai integration: one ``@spec`` runner class.

``rilixai push`` targets this file. :class:`HotpotQARunner` is the entire
sandbox integration — rilixai assembles the metrics calculator, seed
candidate, and per-component feedback from the declarations on ``@spec`` and
the runner's own ``_package_result`` (which emits the paper-style trace).

:func:`build_hotpotqa_spec` is the thin factory the local CLI uses: it loads
its own splits (sized by CLI flags), may inject a scripted agent for tests,
and reuses :class:`HotpotQARunner` so the runtime + scoring match the sandbox
path exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel
from rilixai import spec
from rilixai.adapters import BaseSampleRunner, CallableApplier, SampleRunResult
from rilixai.metrics import BaseMetricsCalculator, FieldConfig
from rilixai.prompt_optimization.models import Sample
from rilixai.prompt_optimization.protocols import ErrorOutput, EvaluationProfile
from rilixai.prompt_optimization.spec import OptimizationContext, PromptOptimizationSpec

from .agent.prompts import hotpotqa_pydantic_agent_seed_candidate
from .agent.types import AgentToolCall, HotpotQAAgentOutput
from .config import HotpotQAConfig
from .data.dataset import HotpotQARecord, load_hotpotqa_paper_split
from .data.eval import exact_match_score, f1_score
from .optimization.feedback import HotpotQAFeedback


_HOTPOTQA_PROFILE_KEY = "hotpotqa"


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
    # No explicit seed: rilixai auto-reads it from the agent's current
    # prompts via the applier's read() at spec-build time.
    # reflection_evidence_mode is kept (rilixai's default is "curated");
    # this agent emits rich trace_evidence the reflection LM should use.
    reflection_evidence_mode="curated_plus_trace",
    # rilixai's default max_concurrency is 8; 4 is the cost-bounded demo value.
    max_concurrency=4,
)
class HotpotQARunner(BaseSampleRunner[HotpotQARecord, HotpotQAAgentOutput]):
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
        # Defer the pydantic-ai-touching import until the runner is built.
        from .agent.agent import HotpotQAPydanticAgent

        self.agent = HotpotQAPydanticAgent(
            model=sandbox_cfg.pydantic_agent_model,
            summarize_model=_bare_openai_model(sandbox_cfg.pydantic_agent_model),
            top_k=sandbox_cfg.retrieve_k,
            max_iters=sandbox_cfg.max_iters,
            temperature=sandbox_cfg.task_temperature,
        )
        super().__init__(
            applier=CallableApplier(
                apply=self.agent.apply_candidate,
                read=lambda: dict(hotpotqa_pydantic_agent_seed_candidate().components),
            )
        )

    async def run_sample(self, record: HotpotQARecord) -> HotpotQAAgentOutput:
        from .agent.retrieval import build_retrieve_k_fn_for_case

        retrieve_k_fn = build_retrieve_k_fn_for_case(record=record, cfg=self.cfg)
        return await self.agent.forward(
            question=record.question,
            paragraphs=record.paragraphs,
            gold_supporting_titles=record.supporting_titles,
            retrieve_k_fn=retrieve_k_fn,
        )

    def _package_result(
        self,
        record: HotpotQARecord,
        output: HotpotQAAgentOutput,
        runtime_kwargs: Mapping[str, Any],
    ) -> SampleRunResult[HotpotQAAgentOutput]:
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
        return SampleRunResult(output=output, run_metrics=run_metrics)

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


# ─── Local-CLI factory ─────────────────────────────────────────────────


def _field_extractor(obj: Any, path: str) -> Any:
    """Resolve a dotted path on a result object (attribute first, item second)."""
    if obj is None or isinstance(obj, ErrorOutput):
        return None
    current: Any = obj
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(segment)
        else:
            current = getattr(current, segment, None)
    return current


def build_hotpotqa_spec(
    *,
    samples_by_split: dict[str, Sequence[Sample]],
    config: HotpotQAConfig | None = None,
    pydantic_agent: Any | None = None,
    model: str | None = None,
    max_concurrency: int = 8,
) -> PromptOptimizationSpec:
    """Assemble a :class:`PromptOptimizationSpec` from explicit splits.

    The local ``python -m hotpotqa.cli`` path uses this: it loads its own
    splits (sized by CLI flags) and may inject a scripted ``pydantic_agent``
    for hermetic tests. Reuses :class:`HotpotQARunner` so the runtime + scoring
    match the sandbox path exactly.
    """
    cfg = config or HotpotQAConfig()
    # The runner reads ``ctx.config`` as a typed HotpotQASandboxConfig; the
    # OptimizationContext field is annotated Mapping[str, Any], so the typed
    # model is a deliberate stand-in (attribute access only).
    ctx = OptimizationContext(
        model=model,
        config=HotpotQASandboxConfig(  # type: ignore[arg-type]
            retrieval_mode=cfg.retrieval_mode,
            retrieve_k=cfg.retrieve_k,
            max_iters=cfg.max_iters,
            pydantic_agent_model=cfg.pydantic_agent_model or "openai:gpt-4.1-mini",
            task_temperature=cfg.pydantic_agent_temperature,
            max_concurrency=max_concurrency,
        ),
    )
    runner = HotpotQARunner(ctx)
    if pydantic_agent is not None:
        runner.agent = pydantic_agent
    # The bridge would attach feedback for the sandbox path; do it by hand here
    # so the local CLI's run_metrics also carry per-component feedback.
    runner.attach_feedback(HotpotQAFeedback())
    metrics = HotpotQAMetrics()
    profile = EvaluationProfile(
        profile_key=_HOTPOTQA_PROFILE_KEY,
        metrics_calculator=metrics,
        field_weights=dict(metrics.field_weights),
    )
    return PromptOptimizationSpec(
        samples_by_split=samples_by_split,
        seed_candidate=hotpotqa_pydantic_agent_seed_candidate(),
        extraction_runtime=runner,
        agent_resolver=lambda **_: (None, None),
        field_extractor=_field_extractor,
        evaluation_profile_resolver=lambda **_: profile,
        name="hotpotqa",
        user_id="__hotpotqa_benchmark__",
        model=model,
        task_type="hotpotqa_pydantic_agent",
        max_concurrency=max_concurrency,
        reflection_evidence_mode="curated_plus_trace",
    )


def _bare_openai_model(pydantic_spec: str) -> str:
    """Strip the provider prefix from a PydanticAI model spec.

    PydanticAI uses ``"openai:gpt-4.1-mini"``; the raw OpenAI summarize call
    wants ``"gpt-4.1-mini"``. Returns the original string when no ``:`` is
    present so already-bare specs pass through unchanged.
    """
    _, separator, model = pydantic_spec.partition(":")
    return model if separator else pydantic_spec


# ─── Trajectory metadata builder (called by the runner's _package_result) ─


def build_agent_run_metrics(
    *,
    record: HotpotQARecord,
    output: HotpotQAAgentOutput,
    agent_kind: str,
    config: HotpotQAConfig,
) -> dict[str, Any]:
    """Translate an agent's tool-call trace into rilixai trajectory metadata.

    Owns only the domain-specific trace evidence (per-hop retrieval reasoning,
    documents-remaining, missing/spurious titles). Per-component feedback flows
    separately through ``@spec(feedback=HotpotQAFeedback)``; the runner merges
    it into ``trace_evidence.per_component_feedback`` in ``_package_result``.
    """
    gold_titles_lower = {t.strip().lower() for t in record.supporting_titles}
    retrieved_titles_lower = {p.title.strip().lower() for p in output.retrieved_paragraphs}
    missing_gold_titles = sorted(
        title for title in record.supporting_titles if title.strip().lower() not in retrieved_titles_lower
    )
    spurious_titles = sorted(
        p.title for p in output.retrieved_paragraphs if p.title.strip().lower() not in gold_titles_lower
    )

    retrieval_span_candidates: dict[str, str] = {
        p.title: _truncate(p.text, config.max_paragraph_chars) for p in output.retrieved_paragraphs
    }

    tool_calls_detail: list[dict[str, Any]] = []
    retrieval_reasoning: list[str] = []
    documents_remaining_per_hop: list[str] = []
    policy_reasoning: list[str] = []
    tool_counts: dict[str, int] = {}
    for step in output.tool_calls:
        tool_calls_detail.append(_tool_call_for_agent_step(step))
        if step.thought:
            policy_reasoning.append(f"Step {step.step_index + 1}: {step.thought}")
        tool_counts[step.tool_name] = tool_counts.get(step.tool_name, 0) + 1
        if step.tool_name == "retrieve_k" or step.tool_name == "search":
            retrieval_reasoning.append(
                f"{step.tool_name} args={step.tool_args}; "
                f"gold titles still missing after this call: {step.gold_titles_remaining_after}"
            )
            documents_remaining_per_hop.append(
                f"Step {step.step_index + 1} ({step.tool_name}): documents remaining to retrieve before this call "
                f"= {step.gold_titles_remaining_before}; after this call = {step.gold_titles_remaining_after}"
            )

    thread_content_parts = [
        f"Question: {record.question}",
        f"Gold answer: {record.answer}",
        f"Gold supporting titles: {list(record.supporting_titles)}",
    ]
    if missing_gold_titles:
        thread_content_parts.append(f"Missing gold titles after retrieval: {missing_gold_titles}")
    if spurious_titles:
        thread_content_parts.append(f"Spurious retrieved titles: {spurious_titles}")
    thread_content = "\n".join(thread_content_parts)

    extraction_reasoning = [
        f"Model answer: {output.answer!r}",
        f"Gold answer: {record.answer!r}",
    ]

    return {
        "tool_counts": {f"hotpotqa_{name}": count for name, count in tool_counts.items()},
        "tool_calls_detail": tool_calls_detail,
        "retrieval_span_candidates": retrieval_span_candidates,
        "thread_content": thread_content,
        "trace_evidence": {
            "retrieval_reasoning": retrieval_reasoning,
            "extraction_reasoning": extraction_reasoning,
            "documents_remaining_per_hop": documents_remaining_per_hop,
            "policy_reasoning": policy_reasoning,
        },
        "hotpotqa": {
            "mode": agent_kind + "_agent",
            "retrieval_mode": config.retrieval_mode,
            "missing_gold_titles": missing_gold_titles,
            "spurious_titles": spurious_titles,
            "num_total_steps": len(output.tool_calls),
        },
    }


def _tool_call_for_agent_step(step: AgentToolCall) -> dict[str, Any]:
    return {
        "tool": f"hotpotqa_{step.tool_name}" if step.tool_name else "hotpotqa_unknown",
        "step_index": step.step_index,
        "args": dict(step.tool_args),
        "return": {
            "observation": step.observation,
            "gold_titles_remaining_before": list(step.gold_titles_remaining_before),
            "gold_titles_remaining_after": list(step.gold_titles_remaining_after),
        },
        "thought": step.thought,
    }


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"
