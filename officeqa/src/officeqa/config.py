"""Constants for the OfficeQA Pro recipe.

Every non-obvious value carries a comment naming its source so a reader can
trace the choice without opening the paper. "The report" below is the
OfficeQA Pro technical report, arXiv:2603.08655 (Databricks, March 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --- Data -------------------------------------------------------------------

# Gated HF dataset. Users request access once on the dataset page and set
# ``HF_TOKEN``. Both the questions CSV and the corpus live in this one repo.
OFFICEQA_HF_REPO = "databricks/officeqa"

# Pinned dataset revision (``sha`` of the HF dataset repo at the time this
# recipe was frozen, 2026-09-10). The split files under ``splits/`` were
# generated from the CSV at this revision; bump both together.
OFFICEQA_DATASET_REVISION = "763a8366abf2a3605c381d53586d844dc60fa756"

# Question files. Only ``officeqa_pro.csv`` (133 ``hard`` rows) is scored.
# ``officeqa_full.csv`` (246 rows = the same 133 hard + 113 easy) is exposed
# behind ``--split dev`` for cheap smoke runs only.
OFFICEQA_PRO_FILE = "officeqa_pro.csv"
OFFICEQA_FULL_FILE = "officeqa_full.csv"

# Corpus subtrees inside the HF repo, and the names they get under the
# per-run ``officeqa_corpus/`` root. The baseline agent works on raw PDFs;
# the parsed text is Databricks' ``ai_parse_document`` output, transformed
# to plain text (report §4.2).
CORPUS_PDF_PATTERN = "treasury_bulletin_pdfs/*"
CORPUS_PARSED_PATTERN = "treasury_bulletins_parsed/transformed/*.txt"
CORPUS_REPRESENTATIONS: dict[str, tuple[str, str]] = {
    # name -> (HF subdirectory, file format label used in the manifest)
    "pdfs": ("treasury_bulletin_pdfs", "pdf"),
    "parsed": ("treasury_bulletins_parsed/transformed", "text"),
}
CORPUS_ROOT_NAME = "officeqa_corpus"
DEFAULT_CORPUS = "pdfs"

# Report §2.1: 697 Treasury Bulletin issues, 1939-2025. Used only as a
# sanity check after download.
EXPECTED_CORPUS_DOCUMENTS = 697


def cache_dir() -> Path:
    """``~/.cache/officeqa/`` unless ``OFFICEQA_CACHE_DIR`` is set."""
    override = os.environ.get("OFFICEQA_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "officeqa"


# --- Agent loop -------------------------------------------------------------

# Report §4.1: "The agent is allowed up to 200 steps per question and uses a
# sliding window that retains the 30 most recent messages, along with a
# reminder message of remaining steps."
MAX_STEPS = 200
WINDOW_SIZE = 30

# Report §4.1 footnote 9: "To prevent out of context errors, tool outputs are
# truncated at 25k characters."
TOOL_OUTPUT_LIMIT = 25_000

# Report §4.1 footnote 9: "All agents are run with a script that allows up to
# 30 retries to restart the agent from its last attempted sample if the agent
# crashes or times out during the run."
MAX_RETRIES = 30

# Report §3.2: "Questions are processed independently within an isolated
# virtual environment so that an agent importing a new package does not
# impact performance on other questions." Each question gets its own
# working directory and its own persistent REPL process.
ISOLATE_PER_QUESTION = True

# Report Appendix D.5 evaluates plurality voting over rollouts. The baseline
# uses a single rollout; ``--n-rollouts > 1`` selects by plurality vote with
# a random tiebreak.
DEFAULT_N_ROLLOUTS = 1

# Harness settings (ours, not from the report). ``--task-timeout`` bounds a
# single question's wall-clock; timeouts score 0 and are not dropped.
DEFAULT_MAX_CONCURRENT = 16
DEFAULT_TASK_TIMEOUT_S = 3600.0

# Baseline tool set: file search + REPL. ``web`` is implemented but not
# registered by default (see ``prompts.SYSTEM_PROMPT_SEED`` for why).
DEFAULT_TOOLS: tuple[str, ...] = ("fs", "repl")
KNOWN_TOOLS: tuple[str, ...] = ("fs", "repl", "web")

# Search arms from report Appendix D.4. Only ``fs`` is implemented; the
# vector-search arms are stubbed and raise ``NotImplementedError``.
DEFAULT_SEARCH = "fs"
KNOWN_SEARCH_ARMS: tuple[str, ...] = ("fs", "vs", "cvs", "fs+cvs")


# --- Models -----------------------------------------------------------------

# Default model choice. Criterion: cheap, fast, and mid-range on the raw-PDF
# full-corpus configuration the baseline uses. Report Table 1 (full corpus,
# PDF, 0.0% threshold):
#     Claude Opus 4.6        48.1%   31.2 min   $4.55 / question
#     GPT-5.4 high           36.1%   13.1 min   $1.79 / question
#     Gemini 3.1 Pro Preview 18.1%   26.4 min   $6.21 / question
# GPT-5.4 is the cheapest and fastest and leaves measurable headroom. Newer
# GPT models (5.5 at $5/$30, 5.6 at $4/$20 per MTok as of 2026-09) are both
# pricier and have no published PDF-mode number, so 5.4 stays the default:
# it is the only GPT string with a report-measured anchor for "is this
# harness broken?". Re-evaluate when a cheaper successor ships.
DEFAULT_MODEL = "gpt-5.4"

# Report Appendix D.2: "Since the GPT 5 model uses medium reasoning and GPT
# 5.1 and 5.4 use no reasoning by default, we set them to high reasoning to
# ensure they're comparable with Anthropic's models."
DEFAULT_REASONING_EFFORT = "high"

# Anthropic IDs for reference runs (first-party IDs, no date suffixes).
ANTHROPIC_REFERENCE_MODELS: tuple[str, ...] = (
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-haiku-4-5",
)

# USD per million tokens: (input, output, cached input). Used when litellm
# has no price for a model string; litellm's table is preferred when it does.
# GPT-5.4 row is report Table 8 (March 2026); Anthropic rows are first-party
# list prices as of 2026-09.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-5.4": (2.50, 15.00, 0.25),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-6": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}

# LLM call settings (ours). Long tool-heavy episodes need a generous
# per-call timeout; retries on transient provider errors are handled by the
# loop's retry budget.
LLM_TIMEOUT_S = 600.0
MAX_OUTPUT_TOKENS = 16_000


# --- Scoring ----------------------------------------------------------------

# Report §3 / Figure 6: correctness is reported at allowable absolute relative
# error thresholds 0.0%, 0.1%, 1.0%, 5.0%; 0.0% is the headline. Values are
# fractions as ``reward.score_answer`` expects (``tolerance=0.001`` == 0.1%).
TOLERANCES: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05)
HEADLINE_TOLERANCE = 0.0


@dataclass(frozen=True)
class RunConfig:
    """Everything a run needs, with the baseline as defaults.

    Constructed by the CLI from flags and by ``run_one`` from keyword
    arguments; serialized verbatim to ``config.json`` in the output dir.
    """

    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    tools: tuple[str, ...] = DEFAULT_TOOLS
    corpus: str = DEFAULT_CORPUS
    search: str = DEFAULT_SEARCH
    n_rollouts: int = DEFAULT_N_ROLLOUTS
    max_steps: int = MAX_STEPS
    window_size: int = WINDOW_SIZE
    tool_output_limit: int = TOOL_OUTPUT_LIMIT
    max_retries: int = MAX_RETRIES
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S
    llm_timeout_s: float = LLM_TIMEOUT_S
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.tools) - set(KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools {sorted(unknown)}; known: {KNOWN_TOOLS}")
        if self.corpus not in CORPUS_REPRESENTATIONS:
            raise ValueError(f"unknown corpus {self.corpus!r}; known: {sorted(CORPUS_REPRESENTATIONS)}")
        if self.search not in KNOWN_SEARCH_ARMS:
            raise ValueError(f"unknown search arm {self.search!r}; known: {KNOWN_SEARCH_ARMS}")
        if self.n_rollouts < 1:
            raise ValueError("n_rollouts must be >= 1")
