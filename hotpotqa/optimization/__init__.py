"""GEPA-facing helpers for the HotpotQA benchmark.

The integration itself — the ``@spec``-decorated :class:`HotpotQARunner`, its
:class:`HotpotQAMetrics`, :func:`build_hotpotqa_spec`, and the trajectory
metadata builder — lives one level up in :mod:`hotpotqa.rilixai_spec`. This
subpackage now holds just the per-component feedback narratives:

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


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "HotpotQAConfig",
    "HotpotQAFeedback",
]
