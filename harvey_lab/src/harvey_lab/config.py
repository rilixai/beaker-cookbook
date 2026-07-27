"""Tunable knobs for the Harvey LAB agent + evaluation.

A single frozen dataclass holds every model / budget / timeout knob so the
CLI and the evaluator thread the same shape. Defaults mirror the Harvey
LAB-AA harness where it makes sense (an ``analyze``-heavy legal benchmark
with long, document-grounded tasks) while staying cheap enough for a smoke
run.
"""

from __future__ import annotations

from dataclasses import dataclass


# The upstream ``harveyai/harvey-labs`` commit the frozen splits were cut
# against (see ``splits/README.md``). Clone the benchmark at this commit for
# reproducible runs — the task set grows over time, and task IDs are paths
# into this exact tree.
HARVEY_LABS_COMMIT = "1da4750171bc5a534960b3d82d15ba7fd2cf653f"

# Global caps used when the frozen ``splits/{val,test}.txt`` were generated.
# Recorded here for documentation; the split files are the source of truth.
VAL_CAP = 100
TEST_CAP = 100


@dataclass(frozen=True)
class HarveyLabConfig:
    """Model + budget configuration for one Harvey LAB run."""

    # The inner legal agent (driven through Stirrup). LiteLLM model spec.
    # Routed through OpenRouter so a single ``OPENROUTER_API_KEY`` covers both
    # the agent and the judge; override with a direct ``openai/…`` (+ that
    # provider's key) if you'd rather call the provider straight.
    task_model: str = "openrouter/openai/gpt-4.1-mini"
    task_temperature: float = 0.0
    # Per-call completion-token cap handed to the Stirrup client (litellm's
    # ``max_tokens``). Must stay within the task model's output limit —
    # gpt-4.1-mini caps completions at 32768, so the default leaves headroom.
    max_output_tokens: int = 16_000

    # The rubric judge. Graded in BATCHES of ``judge_batch_size`` criteria per
    # LLM call rather than one call per criterion — batched verification is an
    # order of magnitude cheaper at LAB's scale (tasks carry ~60 criteria).
    # DeepSeek v4 Flash is a cheap open verifier that stays near frontier
    # graders on LAB. See:
    #   https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents
    #   https://www.appliedcompute.com/case-studies/harvey  (GPT-5 Mini, 4/call)
    judge_model: str = "openrouter/deepseek/deepseek-v4-flash"
    judge_batch_size: int = 8

    # Cap on the Stirrup agent's tool-use loop per task. LAB tasks are long
    # (dozens of documents, ~60 rubric criteria) but a smoke budget stays low;
    # raise it for parity with a full LAB-AA harness.
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
