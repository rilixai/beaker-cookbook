"""Faithful ReAct toolbelt agent for the APEX-Agents benchmark.

A native async ReAct agent that mirrors Mercor Archipelago's
reference harness shape:

* **Tools**:
  * meta tools (``toolbelt_*``, ``todo_write``, ``final_answer``)
  * domain tools over the :class:`WorldFiles` surface
    (``list_files``, ``read_file``, ``read_pdf``,
    ``read_spreadsheet``, ``read_docx``, ``search_files``)
The rilixai component mapping lives in :mod:`apex_agents.rilixai_spec`; this
package stays focused on the agent itself.
"""

from .agent import ApexReActAgent
from .prompts import (
    DEFAULT_RESUM_SUMMARY_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TASK_TEMPLATE,
)
from .types import AgentToolCall, ApexAgentsAgentOutput


__all__ = [
    "DEFAULT_RESUM_SUMMARY_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TASK_TEMPLATE",
    "AgentToolCall",
    "ApexAgentsAgentOutput",
    "ApexReActAgent",
]
