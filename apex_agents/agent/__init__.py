"""Faithful ReAct toolbelt agent for the APEX-Agents benchmark.

A native async ReAct agent that mirrors Mercor Archipelago's
reference harness shape:

* **Tools**:
  * meta tools (``toolbelt_*``, ``todo_write``, ``final_answer``)
  * domain tools over the :class:`WorldFiles` surface
    (``list_files``, ``read_file``, ``read_pdf``,
    ``read_spreadsheet``, ``read_docx``, ``search_files``)
* **Three optimizable components** — rewritten by rilixai's GEPA loop:
  * ``system_prompt`` — tool-use policy
  * ``task_template`` — task framing (Jinja2 ``{{task}}`` placeholder)
  * ``resum_summary_prompt`` — ReSum compaction prompt

The per-component reflection feedback strings live in
:mod:`apex_agents.feedback`.
"""

from .agent import ApexReActAgent
from .prompts import (
    RESUM_SUMMARY_PROMPT_COMPONENT,
    RESUM_SUMMARY_PROMPT_SEED,
    SYSTEM_PROMPT_COMPONENT,
    SYSTEM_PROMPT_SEED,
    TASK_TEMPLATE_COMPONENT,
    TASK_TEMPLATE_SEED,
    apex_agents_seed_candidate,
    load_apex_agents_seed_prompts,
)
from .types import AgentToolCall, ApexAgentsAgentOutput


__all__ = [
    "RESUM_SUMMARY_PROMPT_SEED",
    "SYSTEM_PROMPT_SEED",
    "TASK_TEMPLATE_SEED",
    "RESUM_SUMMARY_PROMPT_COMPONENT",
    "SYSTEM_PROMPT_COMPONENT",
    "TASK_TEMPLATE_COMPONENT",
    "AgentToolCall",
    "ApexAgentsAgentOutput",
    "ApexReActAgent",
    "apex_agents_seed_candidate",
    "load_apex_agents_seed_prompts",
]
