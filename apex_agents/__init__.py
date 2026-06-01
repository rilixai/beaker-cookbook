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
* :mod:`apex_agents.optimization` — GEPA-facing surface (metrics
  judge + per-component feedback); the spec/runner/run-metrics
  builder live in :mod:`apex_agents.rilixai_spec`.
* :mod:`apex_agents.config` — :class:`ApexAgentsConfig` shared by the
  CLI and the runtime.
* :mod:`apex_agents.cli` — local command-line entry point.
* :mod:`apex_agents.sandbox` — rilixai Modal sandbox entry point.
"""

from .agent.prompts import (
    RESUM_SUMMARY_PROMPT_SEED,
    SYSTEM_PROMPT_SEED,
    TASK_TEMPLATE_SEED,
    apex_agents_seed_candidate,
)
from .agent.types import AgentToolCall, ApexAgentsAgentOutput
from .config import ApexAgentsConfig
from .data.dataset import (
    ApexAgentsRecord,
    RubricCriterion,
    cases_from_records,
    load_apex_agents_cases,
    load_apex_agents_records,
    record_to_sample,
)
from .optimization.feedback import ApexAgentsFeedback
from .optimization.metrics import (
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    build_rubric_judge,
    score_rubric,
)
from .rilixai_spec import (
    ApexAgentsMetrics,
    ApexAgentsRunner,
    ApexAgentsSandboxConfig,
    build_apex_agents_run_metrics,
    build_apex_agents_spec,
)


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "RESUM_SUMMARY_PROMPT_SEED",
    "SYSTEM_PROMPT_SEED",
    "TASK_TEMPLATE_SEED",
    "RUBRIC_FIELD",
    "AgentToolCall",
    "ApexAgentsAgentOutput",
    "ApexAgentsConfig",
    "ApexAgentsFeedback",
    "ApexAgentsMetrics",
    "ApexAgentsRecord",
    "ApexAgentsRunner",
    "ApexAgentsSandboxConfig",
    "RubricCriterion",
    "apex_agents_seed_candidate",
    "build_apex_agents_run_metrics",
    "build_apex_agents_spec",
    "build_rubric_judge",
    "cases_from_records",
    "load_apex_agents_cases",
    "load_apex_agents_records",
    "record_to_sample",
    "score_rubric",
]
