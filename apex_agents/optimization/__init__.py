"""GEPA-facing surface for the APEX-Agents benchmark.

Everything the rilixai optimizer consumes lives here:

* :mod:`.runtime` — the async ``ExtractionRuntime`` adapter the
  optimizer invokes per case.
* :mod:`.spec` — assembles the :class:`PromptOptimizationSpec` that
  ``run_optimization_from_spec`` / ``build_adapter_from_spec``
  consume. Also hosts the ``@spec(name="apex-agents")``-decorated
  factory rilixai's Modal sandbox invokes.
* :mod:`.metrics` — LLM-judge rubric scorer + ``MetricsCalculator``.
* :mod:`.feedback` — per-component reflection feedback strings.

A reader who wants to understand "what does GEPA see for APEX-Agents"
can read just this subpackage. The agent internals it composes live
in :mod:`apex_agents.agent`.
"""

from __future__ import annotations

from .feedback import build_apex_per_component_feedback
from .metrics import (
    APEX_AGENTS_FIELD_WEIGHTS,
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    ApexAgentsMetricsCalculator,
    build_apex_agents_field_extractor,
    build_rubric_judge,
    score_rubric,
)
from .runtime import (
    ApexAgentsRunResult,
    build_apex_agents_run_metrics,
    build_apex_agents_runtime,
)
from .spec import build_apex_agents_spec, build_spec


__all__ = [
    "APEX_AGENTS_FIELD_WEIGHTS",
    "DEFAULT_JUDGE_MODEL",
    "RUBRIC_FIELD",
    "ApexAgentsMetricsCalculator",
    "ApexAgentsRunResult",
    "build_apex_agents_field_extractor",
    "build_apex_agents_run_metrics",
    "build_apex_agents_runtime",
    "build_apex_agents_spec",
    "build_apex_per_component_feedback",
    "build_rubric_judge",
    "build_spec",
    "score_rubric",
]
