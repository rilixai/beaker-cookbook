"""GEPA-facing surface for the HotpotQA benchmark.

Everything the rilixai optimizer consumes lives here:

* :mod:`.runtime` — the async ``ExtractionRuntime`` adapter the optimizer
  invokes per case.
* :mod:`.spec` — assembles the :class:`PromptOptimizationSpec` that
  ``run_optimization_from_spec`` and ``build_adapter_from_spec``
  consume.
* :mod:`.metrics` — the :class:`HotpotQAMetricsCalculator` that scores
  per-case outputs against ground truth.
* :mod:`.feedback` — per-component textual feedback the reflection LM
  reads when rewriting ``policy_prompt`` / ``summarize_prompt``.

A reader who wants to understand "what does GEPA see for HotpotQA"
can read just this subpackage. The agent internals it composes live in
:mod:`hotpotqa.agent`.
"""

from __future__ import annotations

from ..config import HotpotQAConfig
from .feedback import (
    AGENT_POLICY_COMPONENT,
    AGENT_SUMMARIZE_COMPONENT,
    build_agent_per_component_feedback,
)
from .metrics import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    HotpotQAFieldConfig,
    HotpotQAMetricsCalculator,
    HotpotQAMetricsResult,
    build_hotpotqa_field_extractor,
)
from .runtime import HotpotQARunResult, build_agent_run_metrics, build_hotpotqa_runtime
from .spec import build_hotpotqa_spec


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "HotpotQAConfig",
    "HotpotQAFieldConfig",
    "HotpotQAMetricsCalculator",
    "HotpotQAMetricsResult",
    "HotpotQARunResult",
    "build_agent_per_component_feedback",
    "build_agent_run_metrics",
    "build_hotpotqa_field_extractor",
    "build_hotpotqa_runtime",
    "build_hotpotqa_spec",
]
