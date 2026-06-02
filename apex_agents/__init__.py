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
* :mod:`apex_agents.rilixai_spec` — the whole GEPA-facing
  integration (``@spec`` runner + ``@spec`` metrics calculator).
* :mod:`apex_agents.metrics` — the LLM rubric judge + the run-metrics
  trajectory builder.
* :mod:`apex_agents.feedback` — optional :class:`ApexAgentsFeedback`
  per-component reflection narratives.
* :mod:`apex_agents.config` — :class:`ApexAgentsConfig` the runner reads.
"""

from .agent.prompts import (
    DEFAULT_RESUM_SUMMARY_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TASK_TEMPLATE,
)
from .agent.types import AgentToolCall, ApexAgentsAgentOutput
from .config import ApexAgentsConfig
from .data.dataset import (
    ApexAgentsRecord,
    RubricCriterion,
    cases_from_records,
    load_apex_agents_cases,
    load_apex_agents_records,
    record_to_case,
)
from .feedback import ApexAgentsFeedback
from .metrics import (
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    build_apex_agents_run_metrics,
    build_rubric_judge,
    score_rubric,
)
from .rilixai_spec import (
    ApexAgentsMetrics,
    ApexAgentsRunner,
    ApexAgentsSandboxConfig,
)


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_RESUM_SUMMARY_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TASK_TEMPLATE",
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
    "build_apex_agents_run_metrics",
    "build_rubric_judge",
    "cases_from_records",
    "load_apex_agents_cases",
    "load_apex_agents_records",
    "record_to_case",
    "score_rubric",
]
