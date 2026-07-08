"""Tunable knobs for the Harvey LAB recipe.

A single frozen dataclass holds every model / budget / timeout knob so
the CLI, the sandbox trigger, and the hosted ``build_spec`` all thread
the same shape. Defaults mirror the Harvey LAB-AA harness where it makes
sense (an ``analyze``-heavy legal benchmark with long, document-grounded
tasks) while staying cheap enough for a smoke run.
"""

from __future__ import annotations

from dataclasses import dataclass


# Pinned upstream commit of ``harveyai/harvey-labs`` the task documents are
# fetched from at runtime (see ``agent/workspace.py``). Pinning a commit keeps
# a spec version reproducible even as the upstream benchmark grows.
HARVEY_LABS_REPO = "harveyai/harvey-labs"
HARVEY_LABS_COMMIT = "f46ef86e4788545622db25dcffa3aebb7a139929"


@dataclass(frozen=True)
class HarveyLabConfig:
    """Model + budget configuration for one Harvey LAB run."""

    # The inner legal agent (driven through Stirrup). LiteLLM model spec.
    task_model: str = "openai/gpt-4.1-mini-2025-04-14"
    task_temperature: float = 0.0
    # Context window budget handed to the Stirrup client, in tokens.
    max_context_tokens: int = 120_000

    # The per-criterion rubric judge. Harvey LAB-AA defaults to
    # ``claude-sonnet``; a cheaper Gemini flash keeps smoke runs affordable
    # and matches the apex recipe's judge default.
    judge_model: str = "gemini/gemini-2.5-flash"

    # Cap on the Stirrup agent's tool-use loop per task. LAB tasks are long
    # (dozens of documents, ~60 rubric criteria) but a smoke budget stays low.
    max_turns: int = 40

    # Per-LLM-call timeout (seconds), shared by the agent model and the judge.
    llm_timeout: float = 120.0

    # Cap on a single document read handed to the agent (bytes of decoded
    # text) so one huge contract can't blow the context window.
    max_document_chars: int = 200_000

    # Cap on the text pulled from ONE deliverable into a judge prompt. LAB
    # criteria are deliverable-scoped, so the judge only ever sees the files a
    # criterion names — this bounds each of those.
    max_deliverable_chars: int = 40_000
