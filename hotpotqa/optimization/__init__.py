"""GEPA-facing helpers for the HotpotQA benchmark.

The integration itself — the ``@spec``-decorated :class:`HotpotQARunner` plus
:class:`HotpotQAMetrics` and :func:`build_hotpotqa_spec` — lives one level up in
:mod:`hotpotqa.rilixai_spec`. This subpackage holds the domain helpers that
runner composes:

* :mod:`.runtime` — :func:`build_agent_run_metrics`, which turns the agent's
  tool-call trace into the optimizer's ``run_metrics`` / ``trace_evidence``.
* :mod:`.feedback` — :class:`HotpotQAFeedback`, whose ``@per_component_feedback``
  methods give the reflection LM per-component narratives.
"""

from __future__ import annotations

from ..config import HotpotQAConfig
from .feedback import (
    AGENT_POLICY_COMPONENT,
    AGENT_SUMMARIZE_COMPONENT,
    HotpotQAFeedback,
)
from .runtime import build_agent_run_metrics


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "HotpotQAConfig",
    "HotpotQAFeedback",
    "build_agent_run_metrics",
]
