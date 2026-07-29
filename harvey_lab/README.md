# Harvey LAB Agent

A **legal knowledge-worker agent** and a rubric evaluation
harness for it. The agent reads a case folder, produces the requested written
deliverables, and is graded criterion-by-criterion against Harvey's public
[Legal Agent Benchmark (LAB)](https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark).

## The agent

The agent plays a **junior lawyer**. For each task it is handed a case folder
and a to-do list, and it must produce the requested written work.

- **Input** — a working directory inside a code execution environment: the
  task's documents staged at `documents/` (contracts, emails, spreadsheets,
  PDFs) plus the task instructions and the exact filenames to produce.
- **Tools** — a **single `code_exec` tool** that runs shell commands in that
  directory, as LAB-AA does. There is no `read_document` / `write_deliverable`
  helper: the agent parses inputs and produces deliverables itself, so real
  `.docx` / `.xlsx` / `.pptx` outputs are on the table (given the tooling — see
  the note below) and the score reflects raw model ability. Vision-capable
  models also get Stirrup's `view_image`.
- **Submission** — `finish` takes a summary plus the **absolute paths** of every
  deliverable and validates each is a real file; nothing outside a successful
  `finish` is graded. `abandon_task_finish` lets the agent give up on a task it
  concludes is impossible.
- **Loop** — the tool-use loop, context management (compaction rather than
  failure at the context limit), and message plumbing come from
  [Stirrup](https://github.com/ArtificialAnalysis/Stirrup), Artificial
  Analysis' agent harness (the one their
  [Harvey LAB-AA leaderboard](https://artificialanalysis.ai/evaluations/harvey-lab-aa)
  runs on). `max_turns` defaults to **200**, matching LAB-AA.
- **Model** — a LiteLLM model string (default `openrouter/deepseek/deepseek-v4-pro`
  run at max reasoning), so any provider LiteLLM routes to works. Reasoning
  budget is `--task-reasoning-effort` (default `xhigh`, the top tier; set `none`
  for a non-reasoning model).
- **Prompts** — `system_prompt` and `task_template` (`agent/prompts.py`), ported
  from [AA's published LAB-AA prompts](https://artificialanalysis.ai/methodology/intelligence-benchmarking#harvey-lab-aa),
  adapted only where they assume AA's specific sandbox.

### Where `code_exec` runs

By default it runs in a temp directory on your machine — no container. The
model's shell runs as your user, so treat it like any script you'd run locally:
prefer a dev box or VM. For real isolation, pass `HarveyLabAgent` an
`exec_provider_factory` returning any other Stirrup `CodeExecToolProvider`
(the framework ships container and remote-sandbox backends); the one
backend-specific touchpoint is `_env_working_dir`.

`pyproject.toml` installs the Python document stack (`python-docx`, `openpyxl`,
`python-pptx`, `pypdf`, `pdfplumber`, `markitdown[all]`) into the same venv the
agent runs from, so those packages are importable from `code_exec`. System
binaries (`pandoc`, `pdftotext`, `soffice`/`libreoffice`) must be installed
separately — see **Install** below.

Code map:

Standard src layout — the importable package lives under `src/harvey_lab/`:

| Path | What it holds |
|---|---|
| `src/harvey_lab/agent/workspace.py` | per-task staging (documents in, deliverables out) + deliverable text extraction |
| `src/harvey_lab/agent/agent.py` | wires `code_exec` + finish/abandon into Stirrup and runs one task |
| `src/harvey_lab/agent/prompts.py` | the two prompts (ported from AA's published LAB-AA prompts) |
| `src/harvey_lab/data/dataset.py` | loads LAB task records from a task tree + reads the splits |
| `src/harvey_lab/data/fetch.py` | on-demand download of just the needed task folders from GitHub |
| `src/harvey_lab/splits/{train,val,test}.txt` | frozen task-id lists (see `src/harvey_lab/splits/README.md`) |
| `src/harvey_lab/evaluation/scoring.py` | grades ONE task: the batched judge + all-pass aggregation |
| `src/harvey_lab/evaluation/run_eval.py` | runs the agent + scoring across the dataset (bounded concurrency) |
| `src/harvey_lab/evaluation/utils.py` | JSON + summary serialization for the CLI |
| `src/harvey_lab/config.py` | model / budget / timeout knobs |
| `src/harvey_lab/cli.py` | run the agent, or run + grade it (`harvey-lab` console command) |

## How the work is graded

Each task ships a rubric of ~60 atomic PASS/FAIL criteria, each with a written
`match_criteria` standard and the deliverable(s) it applies to. A second LLM
acts as a **judge** (`evaluation/scoring.py`): it reads only the deliverable(s)
a criterion names (deliverable-scoped) and returns PASS/FAIL. Rather than one
call per criterion, criteria that share a deliverable scope are graded in
**batches** of `judge_batch_size` (default 8) per call — batched verification
is roughly an order of magnitude cheaper at LAB's scale with near-frontier
agreement (see
[LangChain](https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents)
and [Applied Compute](https://www.appliedcompute.com/case-studies/harvey)). The default judge is `openrouter/deepseek/deepseek-v4-flash`.

Grading is **text-only**, as in LAB-AA: a deliverable is text-extracted (a
`.docx` through python-docx, a `.pptx` through python-pptx, and so on) and the
judge sees that text, the task description, and the criterion's
`match_criteria`. Two LAB-AA rules are enforced:

- **Exact filenames.** A deliverable saved under a near-miss name counts as not
  produced — no fuzzy matching.
- **Partial submissions are still judged.** A criterion fails outright, without
  reaching the judge, only when *none* of its declared deliverables exist. If
  some exist, it is judged with the absent ones marked as such.

Two numbers come out per task:

- `all_pass` (0/1) — `1.0` **iff every** criterion passes, else `0.0`.
- `criterion_pass_rate` — the fraction of criteria passed (LAB-AA's default
  headline metric, and a denser view of the same grading since a single missed
  criterion zeroes `all_pass`).

The evaluation runner (`evaluation/run_eval.py`) averages both across tasks as
`all_pass_rate` / `criterion_pass_rate`. A task whose rubric has no scoreable
criteria is **unscoreable** and excluded from the averages; a task that errors
counts as `0` (a real failure must deflate, not inflate, the metrics). A judge API
or context-window failure aborts grading instead of publishing partial rates.

## The data (and what it is *not*)

Tasks come from the **public** GitHub repo
[`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs) at the pinned
`config.HARVEY_LABS_COMMIT`. The full tree is ~2.7 GB, so by default a run
**fetches only the task folders it needs** (the chosen split, capped by
`--limit`) from GitHub into a local cache — no manual clone. Pass
`--tasks-root` to point at an existing local checkout's `tasks/` dir instead.
Most tasks are tiny (a handful of files); a few diligence data-rooms have
thousands, so downloads run in parallel and **resume** if interrupted
(files already on disk are skipped on the next run). Task directories nest
(larger areas add sub-categories, e.g. `contracts/banking/<slug>`), so a task
ID is the directory path under `tasks/`.
The train / val / test partition is **frozen**: the committed
`splits/{train,val,test}.txt` lists (1560 / 100 / 100 tasks, drawn to follow
the benchmark's natural practice-area distribution) are the source of truth —
see `src/harvey_lab/splits/README.md`. `--split` picks one; `--limit N` runs
just the first N (the lists are ordered so any prefix stays representative).

This is not the [LAB-AA leaderboard](https://artificialanalysis.ai/methodology/intelligence-benchmarking#harvey-lab-aa): that runs on Harvey's **private** 120-task set,
whereas this runs on the **public** tasks with a local (unsandboxed) `code_exec`
and a cheaper batched judge. Treat the scores as a self-contained measurement on
this harness, not a leaderboard-comparable number.

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory:

```bash
cd harvey_lab
uv sync --group dev
```

The agent also needs a few system binaries on `PATH` so it can read and convert
Office/PDF documents. On macOS:

```bash
brew install pandoc poppler
brew install --cask libreoffice
ln -s /Applications/LibreOffice.app/Contents/MacOS/soffice /usr/local/bin/soffice
```

(The `pandoc`, `pdftotext`, and `soffice` binaries are referenced directly in
the task prompt, so they must be available in the agent's shell. `pdfplumber`,
`markitdown[all]`, `python-docx`, `openpyxl`, and `python-pptx` are installed by
`uv sync` from `pyproject.toml`.)

One provider key covers everything — both the agent and the judge default to
OpenRouter routes, so a single key is enough:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Prefer calling providers directly? Override the models with direct LiteLLM
strings and set those providers' keys, e.g.
`--task-model deepseek/deepseek-v4-pro` (`DEEPSEEK_API_KEY`) and
`--judge-model deepseek/deepseek-v4-flash` (`DEEPSEEK_API_KEY`).

## Run

```bash
# Run the agent and dump its deliverables (no grading), on 10 test tasks.
# The 10 task folders are fetched from GitHub on first use and cached.
uv run harvey-lab run \
    --split test --limit 10 \
    --output-dir harvey_lab_run

# Grade everything (reuse saved outputs, run and save if any missing tasks):
uv run harvey-lab evaluate \
    --split test --limit 10 \
    --output-dir harvey_lab_run
```

`run` writes each submitted deliverable **byte-for-byte under its original
filename** plus `run_outputs.json`. `evaluate` uses that manifest as a resume
point: successfully finished or explicitly abandoned tasks are reused only when
their files are still present and their task metadata and source-document
fingerprint is unchanged. Missing, errored, changed, or max-turn-exhausted tasks
are run and saved. It then reloads every selected output from disk before grading
and writes `eval_summary.json` (aggregate `all_pass_rate` /
`criterion_pass_rate` + case counts) and `eval_outputs.json` (per-task results).
Persisted agent outputs remain available for a retry. Use `--rerun` to force
all selected tasks to run again, for example after changing the task model or
agent settings.

See `--help` for all flags (`--split {train,val,test}`, `--limit`,
`--tasks-root`, `--cache-dir`, `--max-concurrency`, `--task-model`,
`--task-reasoning-effort`, `--judge-model`, `--judge-batch-size`,
`--judge-num-retries`, `--shell-timeout`, `--max-turns`, `--no-view-image`,
`--rerun`, …).
