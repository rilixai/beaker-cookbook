"""GEPA-facing surface for the HotpotQA benchmark.

Everything the rilixai optimizer consumes lives here:

* :mod:`.spec` — the class-style ``@spec`` :class:`HotpotQARunner` (the whole
  sandbox integration) plus :func:`build_hotpotqa_spec`, the thin factory the
  local CLI uses. Scoring (:class:`HotpotQAMetrics`, built on
  ``rilixai.metrics.BaseMetricsCalculator``) lives here too.
* :mod:`.runtime` — :func:`build_agent_run_metrics`, which turns the agent's
  tool-call trace into the optimizer's ``run_metrics`` / ``trace_evidence``.
* :mod:`.feedback` — per-component textual feedback the reflection LM reads
  when rewriting ``policy_prompt`` / ``summarize_prompt``.

A reader who wants to understand "what does GEPA see for HotpotQA" can read
just this subpackage. The agent internals it composes live in
:mod:`hotpotqa.agent`.
"""

from __future__ import annotations

from ..config import HotpotQAConfig
from .feedback import (
    AGENT_POLICY_COMPONENT,
    AGENT_SUMMARIZE_COMPONENT,
    build_agent_per_component_feedback,
)
from .runtime import build_agent_run_metrics
from .spec import (
    HotpotQAMetrics,
    HotpotQARunner,
    HotpotQASandboxConfig,
    build_hotpotqa_spec,
)


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "HotpotQAConfig",
    "HotpotQAMetrics",
    "HotpotQARunner",
    "HotpotQASandboxConfig",
    "build_agent_per_component_feedback",
    "build_agent_run_metrics",
    "build_hotpotqa_spec",
]
