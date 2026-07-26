"""APEX-Agents — a ReAct toolbelt agent + a local rubric evaluation.

The Mercor APEX-Agents benchmark (Law / Investment Banking) wrapped
as a native async ReAct agent: meta tools + domain tools over an
on-disk world (zip-extracted per-case files), three prompts
(``system_prompt`` + ``task_template`` + ``resum_summary_prompt``),
and a single metric (``rubric_pass_rate``) scored by an LLM judge
against the per-task rubric.

Layout (each subpackage groups one concern):

* :mod:`apex_agents.agent` — ReAct loop + tools + per-case world
  surface (zip extraction / openpyxl / pypdf / python-docx).
* :mod:`apex_agents.data` — HF dataset loader + world-level splits +
  stratified record cap.
* :mod:`apex_agents.evaluation` — rubric judge, bounded-concurrency
  batch evaluator, JSON report.
* :mod:`apex_agents.config` — :class:`ApexAgentsConfig` shared by the
  CLI and the agent.
* :mod:`apex_agents.cli` — the ``run`` / ``evaluate`` entry point.
"""

from .agent.agent import ApexReActAgent
from .agent.prompts import (
    RESUM_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
    load_apex_agents_prompts,
)
from .agent.types import AgentToolCall, ApexAgentsAgentOutput
from .config import ApexAgentsConfig
from .data.dataset import (
    ApexAgentsRecord,
    RubricCriterion,
    load_apex_agents_records,
)
from .evaluation.local_eval import (
    EvalReport,
    evaluate_agent_on_records,
    evaluate_record,
)
from .evaluation.scoring import (
    DEFAULT_JUDGE_MODEL,
    RUBRIC_FIELD,
    build_rubric_judge,
    score_rubric,
)


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "RESUM_SUMMARY_PROMPT",
    "RUBRIC_FIELD",
    "SYSTEM_PROMPT",
    "TASK_TEMPLATE",
    "AgentToolCall",
    "ApexAgentsAgentOutput",
    "ApexAgentsConfig",
    "ApexAgentsRecord",
    "ApexReActAgent",
    "EvalReport",
    "RubricCriterion",
    "build_rubric_judge",
    "evaluate_agent_on_records",
    "evaluate_record",
    "load_apex_agents_prompts",
    "load_apex_agents_records",
    "score_rubric",
]
