# OfficeQA Pro

A grounded-reasoning agent on [OfficeQA Pro](https://arxiv.org/abs/2603.08655):
133 hard questions over 697 U.S. Treasury Bulletins (1939–2025, ~4.2 GB of PDFs).
The agent gets a filesystem, a Python REPL and a system prompt; it has to find
the right bulletin, extract the right number, and answer inside
`<FINAL_ANSWER>` tags. Scored with the upstream OfficeQA scorer at 0.0% error.

**Status: WIP.** The harness is complete and hermetically tested. The only
model run so far is a 5-question fidelity check (see
[Measured so far](#measured-so-far)); the baselines are still pending, so
the headline cells are empty. See
[What has not been measured yet](#what-has-not-been-measured-yet).

## Prerequisites

1. **Dataset access.** `databricks/officeqa` is gated. Request access at
   https://huggingface.co/datasets/databricks/officeqa, then put a read token in
   `HF_TOKEN` (see `.env.example`).
2. **Disk.** ~4.5 GB in `~/.cache/officeqa/` (override with `OFFICEQA_CACHE_DIR`):
   4.2 GB of PDFs plus 0.5 GB of parsed text. Nothing is copied per question;
   each working directory gets a read-only symlink to the corpus.
3. **OCR toolchain for `--corpus pdfs`** (the default). Without it, upstream
   pilot runs spent hours per question installing PDF libraries. Run

   ```bash
   bash scripts/install-ocr-deps.sh          # apt packages + core Python PDF/OCR packages
   bash scripts/install-ocr-deps.sh --full   # also paddleocr / surya / doctr (several GB)
   ```

   System packages (report Appendix E.3): `tesseract-ocr tesseract-ocr-eng
   libtesseract-dev libleptonica-dev poppler-utils ghostscript libgl1
   libglib2.0-0 ripgrep imagemagick ocrmypdf qpdf pdfgrep`. Python packages:
   `pytesseract easyocr pymupdf opencv-python-headless Pillow pdf2image
   pdfplumber pypdf PyPDF2 pdfminer.six rapidocr-onnxruntime ocrmypdf openpyxl
   camelot-py[base] tesserocr paddlepaddle paddleocr surya-ocr
   python-doctr[torch]`. Per-question virtualenvs are created with
   `--system-site-packages`, so whatever is installed in `.venv` is visible to
   the agent. `--corpus parsed` needs none of this.
4. **A model key** for your `--model` (LiteLLM strings; default `gpt-5.4`, so
   `OPENAI_API_KEY`).

```bash
cd officeqa
uv sync --group dev
cp .env.example .env   # fill in HF_TOKEN and your model key
```

## Reproduce

```bash
# 1. Warm the cache (no model key needed). Downloads are resumable.
uv run officeqa fetch --split test

# 2. Run the baseline on the first 10 test questions.
uv run officeqa run --split test --limit 10 --output-dir runs/test-gpt54

# 3. Resume + score. Clean results are reused, errored/timed-out ones re-run.
uv run officeqa evaluate --split test --output-dir runs/test-gpt54
uv run officeqa evaluate --split test --output-dir runs/test-gpt54 --rerun          # force everything
uv run officeqa evaluate --split test --output-dir runs/test-gpt54 --summary-only   # just aggregate
```

`run` writes `results/<uid>.json` per question (final answer, scores at every
tolerance, full trajectory, tool-call counts, token usage, cost, latency, queue
wait, attempts), `config.json` (flags, uids, dataset revision, corpus manifest,
cpu/RAM readout) and `summary.json`. All flags default to the baseline:

| Flag | Default | Notes |
|---|---|---|
| `--model` | `gpt-5.4` | any LiteLLM model string |
| `--reasoning-effort` | `high` | passed explicitly for GPT-5.x (they default to none) |
| `--tools` | `fs,repl` | add `web` for DuckDuckGo search |
| `--corpus` | `pdfs` | or `parsed` (Databricks `ai_parse_document` text) |
| `--search` | `fs` | `vs`, `cvs`, `fs+cvs` are accepted but raise `NotImplementedError` |
| `--n-rollouts` | `1` | >1 takes a plurality vote over final answers |
| `--max-steps` | `200` | |
| `--window-size` | `30` | recent messages kept besides the pinned system + question |
| `--tool-output-limit` | `25000` | characters per tool result |
| `--max-retries` | `30` | run-wide retry budget for crashes/timeouts |
| `--max-concurrent` | `16` | warned about when > cores in PDF mode |
| `--task-timeout` | `3600` | seconds; a timeout scores 0 and is kept in the aggregate |

Beaker entry point (mirrors `automationbench/`):

```python
from officeqa import load_split, run_one

sample = load_split("train")[0]
result = run_one(sample, model="gpt-5.4", tools=("fs", "repl"), corpus="pdfs")
result.correct  # 0/1 at the 0.0% threshold
result.scores  # {"0.0": ..., "0.001": ..., "0.01": ..., "0.05": ...}
result.trajectory  # full message / tool-call trace
result.tool_calls, result.cost_usd, result.latency_s
```

Each call builds a fresh agent from `officeqa.agent`; in a long-lived
interpreter call `officeqa.reload()` after editing `agent/` to pick the edits
up without restarting.

## The baseline, and why it is under-provisioned

The baseline is deliberately the weakest sensible configuration, so that
everything above it is measurable headroom:

| Constraint in the baseline | Documented headroom (report) |
|---|---|
| **No web search** (`--tools fs,repl`) | 22% of questions need an internet lookup (exchange rates, CPI, …) ⇒ hard ceiling ≈ 78% |
| **Raw PDFs**, not parsed text (`--corpus pdfs`) | 6–20 pp lower than the parsed corpus (Table 1) |
| **grep-only retrieval** (`--search fs`) | up to ~19 pp from adding contextual vector search (Table 6) |
| One rollout | plurality vote over 4 rollouts adds a few pp (Appendix D.5) |

Expected: roughly 20–30%. A baseline near zero means something is broken.

The system prompt is the report's Appendix E.5 prompt verbatim, minus the one
sentence advertising web search (that sentence comes back when `web` is in
`--tools`). The corpus is mounted at `../officeqa_corpus/` relative to the
working directory, exactly as the prompt says; both `pdfs/` and `parsed/` are
listed in the manifest the agent receives, with no hint about which to use.

**No published baseline exists for this agent on the raw-PDF corpus.** Every
custom-agent table in the report uses the parsed corpus; the report's PDF
numbers come from the providers' own CLIs (Codex, Claude Code, Gemini CLI). Our PDF-mode
number is therefore novel, not a reproduction.

### Model choice

`gpt-5.4` at `high` reasoning. From the report's full-corpus PDF table
(Table 1) — the configuration the baseline uses — it is the cheapest, the
fastest, and mid-range rather than at the floor or ceiling:

| Model | Correctness | Latency | Cost/question |
|---|---|---|---|
| Claude Opus 4.6 | 48.1% | 31.2 min | $4.55 |
| **GPT-5.4 high** | **36.1%** | **13.1 min** | **$1.79** |
| Gemini 3.1 Pro Preview | 18.1% | 26.4 min | $6.21 |

Anthropic reference models (`claude-opus-5`, `claude-opus-4-6`,
`claude-sonnet-5`, `claude-haiku-4-5`) are wired with first-party pricing in
`config.py`. Cost is taken from LiteLLM's response when it reports one, else
computed from that table; unknown models record `cost_usd: null`.

## Splits

`train` (93) / `test` (40) partition the 133 Pro questions, stratified on
number of source bulletins (1 / 2 / 3+) × decade of the earliest bulletin,
shuffled with seed 0, allocated by largest remainder and written round-robin
across strata so `--limit N` is a representative prefix. Full procedure and
stratum table: [`src/officeqa/splits/README.md`](src/officeqa/splits/README.md).

**This split is ours.** It is not a published partition; nothing measured on
`test` is comparable to the report, which evaluates on all 133 questions.
Optimize on `train`, report on `test`.

There is also `--split dev`: the 113 `easy` questions of `officeqa_full.csv`,
which are not part of OfficeQA Pro. Use it for cheap smoke runs only.

## Scoring

`src/officeqa/evaluation/reward.py` is copied verbatim from
[databricks/officeqa](https://github.com/databricks/officeqa) (Apache 2.0, see
`ATTRIBUTION.md`). It handles bare numbers (`123.45`), currency
(`$7,046,001.98`) and bracketed lists (`[Massachusetts, 0.866]`), extracts the
last `<FINAL_ANSWER>` block, and takes a relative tolerance. We score every
question at all four tolerances `0.0 / 0.001 / 0.01 / 0.05`; the headline is
**0.0%**. Missing tags, crashes and timeouts all score 0 and stay in the
denominator.

## Agent

- **Loop** (`agent/agent.py`): ≤200 steps; system prompt and question pinned,
  then a 30-message sliding window; a remaining-steps reminder each turn;
  tool results truncated to 25k characters; final answer parsed from the last
  `<FINAL_ANSWER>` block.
- **`fs_search`**: list a directory, list by filename pattern, recursive
  content search (ripgrep when on `PATH`, Python fallback), search one file.
- **`fs_read`**: first N lines, a line slice, or a whole file (refused above
  the output limit; PDFs are refused with a hint to use the REPL).
- **`python_exec`**: one persistent subprocess REPL per question in its own
  working directory and virtualenv; `glob`, `os.listdir`, `os.walk`,
  `os.scandir`, `pathlib` iteration and shell `ls`/`find`/`tree`/`fd` are
  blocked so the agent must search rather than enumerate the corpus.
- **`web_search`** (only with `--tools ...,web`): DuckDuckGo via `ddgs`.

Both tools are confined to the working directory and the corpus symlink;
symlink escapes are rejected.

### Concurrency

Startup logs `os.cpu_count()`, available RAM and `--max-concurrent`, and warns
when concurrency exceeds cores in PDF mode (each worker spawns OCR processes).
Every result records `latency_s` and `queue_wait_s`. The 4-vs-16 measurement
that settles the default has not been run yet (see below).

## Measured so far

One fidelity check (step 6 of the build plan), run on 2026-09-11:

```
officeqa run --split train --limit 5 --corpus parsed --tools fs,repl,web --search fs \
             --model gpt-5.4 --max-concurrent 5
```

| | This harness, `gpt-5.4` (resolved `gpt-5.4-2026-03-05`), n=5 train | Report, GPT-5.4 high, n=133 |
|---|---|---|
| Correctness @0.0% | 3/5 = 60% (same at 0.1/1/5%) | 51.1% |
| Mean latency | 15.2 min (median 13.4; one question ran 45 min and answered on step 199 of 200) | 10.9 min |
| Mean tool calls | 144.4 (median 176; 432 `fs_search`, 121 `fs_read`, 119 `python_exec`, 50 `web_search`) | 104.8 |
| Cost / question | $5.44 (total $27.20) | $6.13 |

All 5 finished `answered` on the first attempt (no retries, timeouts, or
errors). With n=5 the binomial standard error on correctness is ±22 points, so
this is directional only: score, cost and tool-call volume are in the report's
range, and the shape (a few cheap questions, a long tail of 180–280-call
searches) is consistent with GPT-5.4 having the highest tool-call count in the
report's table. `web_search` hit provider rate limits (Brave/Google 429) but
fell through to other backends.

## What has not been measured yet

Steps 1–5 of the build plan (harness, data, splits, scorer, agent, tests) cost
nothing and are done. The following spend money and are **pending**:

| Step | What it produces for this README |
|---|---|
| Probe (10 train questions at `--max-concurrent 4` and `16`) | the concurrency default and whether the agent opens `parsed/` when it appears in the manifest |
| Baseline on `train` | score, wall-clock, dollars; count of questions blocked by the missing web tool |
| Baseline on `test` | headline numbers, baseline model + one Anthropic reference model |

Report reference (custom agent, parsed corpus, file search, 0.0% threshold;
Appendix D.2 Table 4):

| Model | Correctness | Latency | Tool calls | Cost/question |
|---|---|---|---|---|
| Claude Opus 4.6 | 57.1% | 5.3 min | 53.3 | $8.61 |
| Claude Sonnet 4.6 | 51.9% | 5.4 | 69.2 | $6.45 |
| GPT-5.4 high | 51.1% | 10.9 | 104.8 | $6.13 |
| Gemini 3.1 Pro Preview | 42.9% | 2.6 | 25.7 | $1.13 |
| Claude Haiku 4.5 | 33.8% | 5.7 | 73.9 | $2.84 |

## The parsed corpus format

`parsed/*.txt` is the `treasury_bulletins_parsed/transformed/` output of
upstream `transform_parsed_files.py`: page text in reading order with tables
rendered as **Markdown pipe tables**, one row per line, header row followed by
a `|---|` separator. Multi-level column headers are flattened into a single
header cell joined with ` > `, e.g.

```
| Fiscal year or month | Total on-budget and off-budget results > Total receipts (1) | ... |
```

Footnote markers stay inline as `(1)`. There is no HTML.

## Tests

```bash
uv run ruff check && uv run ruff format --check && uv run python -m mypy
uv run python -m pytest -q
```

Hermetic (no network, no model calls): synthetic three-document corpus and
three synthetic questions in `tests/conftest.py`, scripted model client.
Covers leakage (no ground truth or source basename in any serialized
trajectory or model request), split invariants and deterministic regeneration,
all three answer shapes at all four tolerances, missing-tag and timeout
scoring, step cap, window size, output truncation, retry budget and resume,
REPL sandbox restrictions, filesystem confinement, and the corpus manifest.
Set `OFFICEQA_PRO_CSV=/path/to/officeqa_pro.csv` to also check that the frozen
split regenerates byte-for-byte from the real CSV.

## Sources

- OfficeQA Pro technical report — https://arxiv.org/abs/2603.08655
- Upstream repo — https://github.com/databricks/officeqa (Apache 2.0 code, CC-BY-SA-4.0 data)
- Dataset — https://huggingface.co/datasets/databricks/officeqa (gated); pinned revision in `config.py`
