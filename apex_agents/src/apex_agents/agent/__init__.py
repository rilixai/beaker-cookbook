"""Faithful ReAct toolbelt agent for the APEX-Agents benchmark.

A native async ReAct agent that mirrors Mercor Archipelago's
reference harness shape:

* **Tools**:
  * meta tools (``toolbelt_*``, ``todo_write``, ``final_answer``)
  * domain tools over the :class:`WorldFiles` surface
    (``list_files``, ``read_file``, ``read_pdf``,
    ``read_spreadsheet``, ``read_docx``, ``search_files``)
* **Three prompts** (``agent/prompts.py``):
  * ``system_prompt`` — tool-use policy
  * ``task_template`` — task framing (``{{task}}`` placeholder)
  * ``resum_summary_prompt`` — ReSum compaction prompt
"""

from .agent import ApexReActAgent
from .prompts import (
    RESUM_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
    load_apex_agents_prompts,
)
from .types import AgentToolCall, ApexAgentsAgentOutput


__all__ = [
    "RESUM_SUMMARY_PROMPT",
    "SYSTEM_PROMPT",
    "TASK_TEMPLATE",
    "AgentToolCall",
    "ApexAgentsAgentOutput",
    "ApexReActAgent",
    "load_apex_agents_prompts",
]
