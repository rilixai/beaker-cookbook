"""GEPA-facing surface for the HotpotQA benchmark.

Everything the rilixai optimizer consumes lives here:

* :mod:`.runtime` — the async ``RunCase`` adapter the optimizer
  invokes per case.
* :mod:`.spec` — assembles the :class:`~rilixai.Spec` that the hosted
  optimizer and ``validate_spec`` consume.
* :mod:`.metrics` — the :class:`HotpotQAScorer` (:class:`~rilixai.CaseScorer`)
  that scores per-case outputs against ground truth.
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
from .local_eval import LocalEvalReport, evaluate_targets_on_cases, run_local_evaluation
from .metrics import (
    ANSWER_F1_FIELD,
    ANSWER_FIELD,
    HOTPOTQA_FIELD_WEIGHTS,
    SUPPORTING_TITLES_RECALL_FIELD,
    HotpotQAScorer,
)
from .runtime import build_agent_run_metrics, build_hotpotqa_run_case
from .spec import build_hotpotqa_spec


__all__ = [
    "AGENT_POLICY_COMPONENT",
    "AGENT_SUMMARIZE_COMPONENT",
    "ANSWER_F1_FIELD",
    "ANSWER_FIELD",
    "HOTPOTQA_FIELD_WEIGHTS",
    "SUPPORTING_TITLES_RECALL_FIELD",
    "HotpotQAConfig",
    "HotpotQAScorer",
    "LocalEvalReport",
    "build_agent_per_component_feedback",
    "build_agent_run_metrics",
    "build_hotpotqa_run_case",
    "build_hotpotqa_spec",
    "evaluate_targets_on_cases",
    "run_local_evaluation",
]
