"""GEPA-facing surface for the APEX-Agents benchmark.

Everything the rilixai optimizer consumes lives here:

* :mod:`.runtime` — the async ``run_case`` adapter the optimizer
  invokes per :class:`~rilixai.Case`.
* :mod:`.spec` — assembles the rilixai :class:`~rilixai.Spec` the
  optimizer consumes. Also hosts the ``@spec(name="apex-agents")``-
  decorated factory rilixai's sandbox invokes.
* :mod:`.metrics` — LLM-judge rubric scorer + :class:`ApexAgentsScorer`.
* :mod:`.local_eval` — SDK-only single-candidate evaluation loop.
* :mod:`.feedback` — per-component reflection feedback strings.

A reader who wants to understand "what does GEPA see for APEX-Agents"
can read just this subpackage. The agent internals it composes live
in :mod:`apex_agents.agent`.
"""

from __future__ import annotations

from .feedback import build_apex_per_component_feedback
from .local_eval import LocalEvalReport, evaluate_targets_on_cases, run_local_evaluation
from .metrics import (
    APEX_AGENTS_FIELD_WEIGHTS,
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    ApexAgentsScorer,
    build_rubric_judge,
    score_rubric,
)
from .runtime import (
    build_apex_agents_run_case,
    build_apex_agents_run_metrics,
)
from .spec import build_apex_agents_spec, build_spec


__all__ = [
    "APEX_AGENTS_FIELD_WEIGHTS",
    "DEFAULT_JUDGE_MODEL",
    "RUBRIC_FIELD",
    "ApexAgentsScorer",
    "LocalEvalReport",
    "build_apex_agents_run_case",
    "build_apex_agents_run_metrics",
    "build_apex_agents_spec",
    "build_apex_per_component_feedback",
    "build_rubric_judge",
    "build_spec",
    "evaluate_targets_on_cases",
    "run_local_evaluation",
    "score_rubric",
]
