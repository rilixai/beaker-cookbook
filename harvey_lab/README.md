# Harvey LAB Agent

A **legal knowledge-worker agent** and a rubric evaluation harness for it. The
agent reads a case folder, produces the requested written deliverables, and is
graded criterion-by-criterion against Harvey's public
[Legal Agent Benchmark (LAB)](https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark).

```bash
cd harvey_lab && uv sync --group dev
export OPENAI_API_KEY=...                  # GPT-6 Luna task agent
export OPENROUTER_API_KEY=...              # GLM 5.3 Flash judge
uv run harvey-lab evaluate --split test --limit 5 --output-dir harvey_lab_run
```

## The agent

The agent plays a **junior lawyer**: a case folder in, the requested written
work out.

- **In / out** — the task's documents (contracts, emails, spreadsheets, PDFs)
  staged at `documents/` inside a code execution environment, plus instructions
  and the exact filenames to produce; out come those files.
- **Tools** — a **single `code_exec` tool**, as LAB-AA does: no `read_document`
  / `write_deliverable` helpers, so the agent parses inputs and builds
  deliverables itself and the score reflects raw model ability (real `.docx` /
  `.xlsx` / `.pptx` outputs are on the table). Vision-capable models also get
  Stirrup's `view_image`. **The shell is unsandboxed — see [Notes](#notes).**
- **Submission** — `finish` takes a summary plus the **absolute paths** of every
  deliverable and validates each is a real file; nothing outside a successful
  `finish` is graded. `abandon_task_finish` gives up on an impossible task.
- **Loop + model** — tool-use loop, context compaction and LLM routing come from
  [Stirrup](https://github.com/ArtificialAnalysis/Stirrup), AA's harness. Any
  LiteLLM model string works (default `openai/gpt-6-luna` at
  `--task-reasoning-effort medium`, with a 1M context window and 128k output cap);
  `--max-turns` defaults to **200**, as LAB-AA.
- **Prompts** — `system_prompt` and `task_template` (`agent/prompts.py`), ported
  from AA's published LAB-AA prompts, adapted where they assume AA's sandbox.

Standard src layout, under `src/harvey_lab/`:

```
agent/       workspace.py (stage documents in, pull deliverables out)
             agent.py (wires code_exec + finish into Stirrup, runs one task)
             prompts.py
data/        dataset.py (task records + splits), fetch.py (on-demand download)
splits/      frozen {train,test}.txt task-id lists  → splits/README.md
evaluation/  scoring.py (batched judge, one task), run_eval.py (all tasks)
cli.py       `harvey-lab` console command; config.py: every model/budget knob
```

## How the work is graded

Each task ships a rubric of ~60 atomic PASS/FAIL criteria, each with a written
`match_criteria` standard and the deliverable(s) it applies to. A second LLM
acts as a **judge** (`evaluation/scoring.py`, default
`openrouter/z-ai/glm-5.3-flash`): it reads only the deliverable(s) a
criterion names — the text extracted from it, as LAB-AA grades text only — and
returns PASS/FAIL. Criteria sharing a deliverable scope are graded in
**batches** of `--judge-batch-size` (default 8) rather than one call per
criterion, an order of magnitude cheaper at LAB's scale
([why](https://www.langchain.com/blog/designing-efficient-verifiers-for-legal-agents)).

Two numbers come out per task, and `evaluation/run_eval.py` averages both:

- `all_pass_rate` — the share of tasks where **every** criterion passed.
- `criterion_pass_rate` — the fraction of criteria passed (LAB-AA's headline
  metric, and a denser view of the same grading since one missed criterion
  zeroes `all_pass`).

Filenames must match **exactly** — a near-miss name counts as not produced. A
criterion skips the judge and fails outright only when *none* of its
deliverables exist. An errored task counts as `0`; a task with no scoreable
criteria is *unscoreable* and excluded from the averages.

## The data

Tasks come from the **public** [`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs)
repo at the pinned `config.HARVEY_LABS_COMMIT`. The full tree is ~2.7 GB, so a
run **fetches only the task folders it needs** into a local cache, in parallel
and resumably — no manual clone (`--tasks-root` uses an existing checkout's
`tasks/` instead). One task is a `task.json` (`instructions`, `deliverables`,
`criteria`) plus a `documents/` tree, and its ID is its path under `tasks/`,
e.g. `contracts/banking/<slug>`.

The train / test partition is **frozen**: the committed
`splits/{train,test}.txt` lists (1660 / 100 tasks) are the source of
truth — see [`src/harvey_lab/splits/README.md`](src/harvey_lab/splits/README.md).
`--split` picks one; `--limit N` runs the first N. Prefixes do not preserve the
full distribution; use the Beaker sampler below for proportional training samples.

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory:

```bash
cd harvey_lab
uv sync --group dev
export OPENAI_API_KEY=...             # GPT-6 Luna task agent
export OPENROUTER_API_KEY=...         # GLM 5.3 Flash judge
export GITHUB_TOKEN=ghp_...           # optional: raises GitHub's 60 req/hour
                                      # API limit used by task fetching
```

`uv sync` installs the Python document stack (`python-docx`, `openpyxl`,
`python-pptx`, `pypdf`, `pdfplumber`, `markitdown[all]`) into the venv the agent
shells out from. `pandoc`, `pdftotext` and `soffice` are named in the task
prompt, so they must be on `PATH` too:

```bash
brew install pandoc poppler && brew install --cask libreoffice   # macOS
ln -s /Applications/LibreOffice.app/Contents/MacOS/soffice /usr/local/bin/soffice
# Linux: pandoc, poppler-utils (pdftotext), libreoffice (soffice)
```

## Run

```bash
# Optional: warm the task cache up front (no model key needed), so `run` /
# `evaluate` never wait on GitHub — one LAB task can be a ~3k-file data room.
uv run harvey-lab fetch --split test --limit 10

# Run the agent and save its deliverables (no grading):
uv run harvey-lab run --split test --limit 10 --output-dir harvey_lab_run

# Run whatever is missing, then grade everything:
uv run harvey-lab evaluate --split test --limit 10 --output-dir harvey_lab_run
```

`run` writes each submitted deliverable **byte-for-byte under its original
filename** plus `run_outputs.json`. `evaluate` treats that manifest as a resume
point — a task that finished or abandoned cleanly with its files and fingerprint
intact is reused, anything else is re-run (`--rerun` forces all) — then writes
`eval_summary.json` (aggregates + case counts) and `eval_outputs.json` (per
task). See `--help` for every flag.

## Beaker integration

Run these commands from `harvey_lab/` after `uv sync --group dev --locked`.
The `harvey-lab` integration in `.beaker/beaker.yaml` uses the existing
`harvey-lab-agent` Beaker agent and optimizes `src/harvey_lab/agent/` for
`criterion_pass_rate`; it also reports `all_pass` and per-criterion feedback.
Setup validates the pinned task instructions and rubric, then stages only
source documents for candidate execution.

```bash
# Inspect the default selection and practice-area counts without downloading tasks:
uv run python .beaker/upload_splits.py --dry-run

# Validate two real tasks without model calls or an upload:
uv run python .beaker/upload_splits.py --smoke-only --train-limit 1 --test-limit 1

# Upload and validate 100 train / all 100 test tasks:
uv run python .beaker/upload_splits.py --train-limit 100 --seed 0 --name harvey-lab-stratified-100
```

The helper uses temporary JSONL files and prints an immutable `name@revision`.
Uploads always keep the frozen 100-task test set unchanged. Training tasks are
sampled without replacement from the frozen 1660-task training pool. For each
of the 26 practice areas, the helper allocates `train_limit * test_area_count / 100`
slots, rounds down, then assigns remaining slots to the largest fractional
remainders (ties use alphabetical area order). It samples uniformly within each
area with `--seed` (default 0). Input list order does not affect the sample.
At the default 100 training tasks, every area has exactly the same count as test;
at other sizes each quota differs from its proportional target by less than one
task. Small samples can omit rare areas. If an area has too few training tasks,
selection fails before downloads instead of changing the quotas.

`--dry-run` prints the selected IDs, seed, and per-area counts and shares. The
uploaded manifest records the same information. Use one seed and dataset revision
across comparisons; do not select the seed using evaluation scores. Matching
practice areas does not guarantee matching document sizes, work types, difficulty,
or independent document families. The default includes diligence tasks and can
require large downloads.

Use `--tasks-root /path/to/tasks` to reuse the pinned corpus, `--train-limit`
for another training size, or `--full` to bypass sampling and use all 1660 train /
100 test tasks (their proportions are not identical). Use a distinct dataset name
when changing the sample. `--test-limit` is allowed only with `--smoke-only` for
small structural checks; these checks are not representative evaluations. Smoke
checks structure, not agent or judge quality.

The integration declares its required hosted judge in `.beaker/beaker.yaml`:

```yaml
requires_scorer_model: true
scorer_model: openrouter:z-ai/glm-5.3-flash
```

These fields belong under `integrations.harvey-lab`. Beaker records the judge
in the immutable integration build and checks provider credentials before launch.
To start a hosted run from the pushed integration branch, use the exact dataset
revision printed by the helper:

```bash
uv run beaker run trigger --integration-id harvey-lab --agent harvey-lab-agent \
  --dataset 'DATASET_NAME@REVISION'
```

Ordinary runs start with GPT-6 Luna at medium reasoning and a 1M context window.
The hosted judge comes from the selected integration's YAML, independently of the
standalone judge default. Launch-time judge overrides and `config_defaults.scorer_model`
are rejected. To change judges, update the integration YAML and push it before
launching from that branch so Beaker builds the new configuration. Use a fresh baseline
on both splits when changing judges; scores are not directly comparable with the
earlier DeepSeek-judged runs. Before a full optimization, evaluate a few training
cases to check judge output and latency. GLM's Beaker catalog default is max reasoning,
so its Flash name alone does not guarantee faster grading.

Selected-model runs use the Beaker gateway, omit temperature and output-token limits, and default
to a 1,000,000-token history window. Set the optional hosted agent variable
`HARVEY_BEAKER_CONTEXT_WINDOW_TOKENS` to the selected model's actual capacity
with `beaker agent env set`; local shell values are not forwarded automatically.
Setup rejects invalid values before downloads or case execution. Stirrup summarizes
at 70% of the window, so even 1M does not retain the large diligence data rooms.
Use separate runs for models that need different context settings.

Beaker grades four criteria per judge batch, versus eight in the standalone
harness. Scores across those batching settings are not directly interchangeable.
Keep the judge and batching fixed across baseline and candidates. Incomplete or
invalid judge responses are retried; exhausted retries raise an error instead
of assigning false failures. Candidate LiteLLM calls are traced separately from
judge calls; separate Stirrup tool spans are not included.

The hosted builder installs this project's `pyproject.toml` automatically and
supplies the managed Beaker packages. `pip_install_from` is unnecessary, and
`pip_install` must not include `beaker-sdk`. The configured image also installs
Pandoc, Poppler, and LibreOffice.

## Tests

```bash
uv run pytest -q
```

Hermetic — a scripted Stirrup client over the local shell backend, a stub judge,
and a fixture task tree: no network, no spend.

## Notes

- **Where `code_exec` runs.** By default: a temp directory on your machine, no
  container. The model's shell runs as your user, so treat it like any script
  you'd run locally — prefer a dev box or VM. For real isolation pass
  `HarveyLabAgent` an `exec_provider_factory` returning another Stirrup
  `CodeExecToolProvider` (see `agent/agent.py`).
- **This is not the [LAB-AA leaderboard](https://artificialanalysis.ai/evaluations/harvey-lab-aa).**
  That runs on Harvey's **private** 120-task set; this runs the **public** tasks
  with a local unsandboxed `code_exec` and a cheaper batched judge. Treat scores
  as a self-contained measurement on this harness, not a comparable number.
- To call providers directly instead of OpenRouter, pass direct LiteLLM strings
  and set those keys: `--task-model deepseek/deepseek-v4-pro` + `DEEPSEEK_API_KEY`.
- Reference: [AA's LAB-AA methodology and prompts](https://artificialanalysis.ai/methodology/intelligence-benchmarking#harvey-lab-aa),
  [Applied Compute on batched Harvey verification](https://www.appliedcompute.com/case-studies/harvey).
- TODO(owner): expected `criterion_pass_rate`, wall-clock, and $ cost for the
  canonical `evaluate --split test --limit 10` run at the default models.
