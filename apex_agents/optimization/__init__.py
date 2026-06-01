"""GEPA-facing helpers for the APEX-Agents benchmark.

The integration itself — the ``@spec``-decorated :class:`ApexAgentsRunner`
plus :class:`ApexAgentsMetrics` and :func:`build_apex_agents_spec` — lives one
level up in :mod:`apex_agents.rilixai_spec`. This subpackage holds the domain
helpers that runner composes:

* :mod:`.runtime` — :func:`build_apex_agents_run_metrics`, the trajectory
  metadata builder.
* :mod:`.metrics` — the LLM rubric judge (:func:`build_rubric_judge`,
  :func:`score_rubric`, :func:`coerce_pass_rate`).
* :mod:`.feedback` — :class:`ApexAgentsFeedback`, whose
  ``@per_component_feedback`` methods give the reflection LM per-component
  narratives.
"""

from __future__ import annotations

from .feedback import ApexAgentsFeedback
from .metrics import (
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    build_rubric_judge,
    coerce_pass_rate,
    score_rubric,
)
from .runtime import build_apex_agents_run_metrics


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "RUBRIC_FIELD",
    "ApexAgentsFeedback",
    "build_apex_agents_run_metrics",
    "build_rubric_judge",
    "coerce_pass_rate",
    "score_rubric",
]
