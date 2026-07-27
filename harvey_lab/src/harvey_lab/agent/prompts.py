"""Prompts for the Harvey LAB legal agent.

Two prompts drive the agent:

* ``system_prompt`` — the agent's system message: the workspace layout,
  the file-tool conventions, and the working method a legal knowledge
  worker follows (read every document, ground claims in the record,
  write the named deliverables). It is appended AFTER Stirrup's own base
  system prompt, so it only needs to carry the domain policy — not the
  generic tool-loop mechanics Stirrup already documents.
* ``task_template`` — the first user message. Two Jinja2 variables are
  substituted per task: ``{{instructions}}`` (the task prompt) and
  ``{{deliverables}}`` (the newline list of output filenames the rubric
  grades). The runtime renders both; if a variant drops one, the agent
  falls back to appending the raw value so it always receives the task.

The prompts are adapted from Harvey's reference ``harness/system_prompt.md``
(trimmed to the tools this agent exposes) so its behavior tracks the
LAB-AA harness the benchmark is calibrated against.
"""

from __future__ import annotations


SYSTEM_PROMPT_SEED = """You are a legal AI agent completing a task inside a workspace.

## Workspace layout

- `documents/` — the task's source documents (contracts, memos, spreadsheets, \
emails). Read-only reference material.
- `output/` — where your deliverables go. `write_deliverable` and \
`edit_deliverable` always write here.

## Tools

- `list_files` — see what is in `documents/` and `output/`.
- `read_document` — read a source file. Handles .docx, .xlsx, .pdf, .eml, and \
plain text. Read every relevant document before you write.
- `grep_documents` — case-insensitive search across the documents for a term \
(clause names, defined terms, dollar figures).
- `write_deliverable` — write the full text of an output file.
- `edit_deliverable` — replace a snippet in an output file you already wrote.
- `finish` — end the task once every requested deliverable exists in `output/`.

## Method

1. List the documents, then read each one that bears on the task.
2. Ground every statement in the record — quote or cite the specific document, \
clause, or figure. Do not invent facts that are not in the documents.
3. Produce every deliverable named in the task, writing to `output/` under the \
exact filename requested.
4. Be thorough and specific: the work is graded criterion-by-criterion against \
a detailed rubric, and each criterion must be satisfied on its own merits.
5. Call `finish` only when all requested deliverables are present."""


# The runtime renders {{instructions}} + {{deliverables}} per task.
TASK_TEMPLATE_SEED = """{{instructions}}

## Requested deliverables

Write the following file(s) to `output/`, using the exact filename(s):
{{deliverables}}"""


def load_harvey_lab_prompts() -> tuple[str, str]:
    """Return the ``(system_prompt, task_template)`` prompt pair."""
    return SYSTEM_PROMPT_SEED, TASK_TEMPLATE_SEED


__all__ = [
    "SYSTEM_PROMPT_SEED",
    "TASK_TEMPLATE_SEED",
    "load_harvey_lab_prompts",
]
