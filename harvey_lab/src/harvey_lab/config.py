"""Tunable knobs for the Harvey LAB agent + evaluation.

A single frozen dataclass holds every model / budget / timeout knob so the
CLI and the evaluator thread the same shape. Defaults track Artificial
Analysis' published Harvey LAB-AA harness settings (200 turns, a single
``code_exec`` tool, a 20-minute per-command timeout) so the agent side of
this recipe is comparable to the leaderboard; the grading side deliberately
diverges (batched cheap judge) and is documented in the README.
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

    # Cap on the Stirrup agent's tool-use loop per task. LAB-AA gives agents
    # up to 200 turns per task; Harvey's own harness defaults to the same.
    max_turns: int = 200

    # Per-LLM-call timeout (seconds), shared by the agent model and the judge.
    llm_timeout: float = 120.0

    # Retries for a failed judge call. Stirrup owns task-model retries.
    judge_num_retries: int = 8

    # ─── execution environment ────────────────────────────────────────
    # The agent's single ``code_exec`` tool runs in a temp directory on THIS
    # machine (Stirrup's local backend). No isolation: the model runs arbitrary
    # shell commands as you — see the README before pointing it at real tasks.
    #
    # NOTE: LAB-AA itself runs `code_exec` inside a remote sandbox, and Harvey's
    # own harness uses a container. This recipe deliberately stays local so it
    # runs with zero setup. Swapping in a sandboxed backend is a self-contained
    # extension: pass an ``exec_provider_factory`` to ``HarveyLabAgent`` that
    # returns any Stirrup ``CodeExecToolProvider`` (the framework ships
    # container- and remote-sandbox backends) — nothing else here needs to
    # change, since the agent only relies on that provider's read/write/exists
    # surface.

    # Per-``code_exec``-command wall-clock timeout (seconds). LAB-AA terminates
    # individual shell commands after 20 minutes.
    shell_timeout_s: int = 1_200

    # Give vision-capable models Stirrup's ``view_image`` tool, which reads an
    # image file out of the execution environment as native image tokens
    # (LAB-AA does this for models that support vision).
    enable_view_image: bool = True

    # Cap on the text pulled from ONE deliverable into a judge prompt. LAB
    # criteria are deliverable-scoped, so the judge only ever sees the files a
    # criterion names — this bounds each of those.
    max_deliverable_chars: int = 40_000
