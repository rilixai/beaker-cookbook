"""APEX-Agents — faithful ReAct toolbelt agent + rilixai GEPA optimization.

The Mercor APEX-Agents benchmark (Law / Investment Banking) wrapped
as a native async ReAct agent: meta tools + domain tools over an
on-disk world (zip-extracted per-case files), three optimizable
prompts (``system_prompt`` + ``task_template`` +
``resum_summary_prompt``), and a single optimization field
(``rubric_pass_rate``) scored by an LLM judge against the per-task
rubric.

Layout (each subpackage groups one concern):

* :mod:`apex_agents.agent` — ReAct loop + tools + per-case world
  surface (zip extraction / openpyxl / pypdf / python-docx).
* :mod:`apex_agents.data` — HF dataset loader + world-level k-fold +
  stratified train cap.
* :mod:`apex_agents.optimization` — GEPA-facing surface (spec,
  runtime adapter, metrics, per-component feedback).
* :mod:`apex_agents.config` — :class:`ApexAgentsConfig` shared by the
  CLI and the runtime.
* :mod:`apex_agents.cli` — local command-line entry point.
* :mod:`apex_agents.sandbox` — rilixai Modal sandbox entry point.
"""

from .agent.prompts import (
    RESUM_SUMMARY_PROMPT_SEED,
    SYSTEM_PROMPT_SEED,
    TASK_TEMPLATE_SEED,
    apex_agents_seed_targets,
)
from .agent.types import AgentToolCall, ApexAgentsAgentOutput
from .config import ApexAgentsConfig
from .data.dataset import (
    ApexAgentsDataLoader,
    ApexAgentsRecord,
    RubricCriterion,
    cases_from_records,
    load_apex_agents_cases,
    load_apex_agents_records,
    record_to_case,
)
from .optimization.feedback import build_apex_per_component_feedback
from .optimization.metrics import (
    APEX_AGENTS_FIELD_WEIGHTS,
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    ApexAgentsScorer,
)
from .optimization.runtime import build_apex_agents_run_case
from .optimization.spec import build_apex_agents_spec


__all__ = [
    "APEX_AGENTS_FIELD_WEIGHTS",
    "DEFAULT_JUDGE_MODEL",
    "RESUM_SUMMARY_PROMPT_SEED",
    "SYSTEM_PROMPT_SEED",
    "TASK_TEMPLATE_SEED",
    "RUBRIC_FIELD",
    "AgentToolCall",
    "ApexAgentsAgentOutput",
    "ApexAgentsConfig",
    "ApexAgentsDataLoader",
    "ApexAgentsRecord",
    "ApexAgentsScorer",
    "RubricCriterion",
    "apex_agents_seed_targets",
    "build_apex_agents_run_case",
    "build_apex_agents_spec",
    "build_apex_per_component_feedback",
    "cases_from_records",
    "load_apex_agents_cases",
    "load_apex_agents_records",
    "record_to_case",
]
