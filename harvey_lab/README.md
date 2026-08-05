# Harvey LAB Agent

A **legal knowledge-worker agent** and a rubric evaluation harness for it. The
agent reads a case folder, produces the requested written deliverables, and is
graded criterion-by-criterion against Harvey's public
[Legal Agent Benchmark (LAB)](https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark).

```bash
cd harvey_lab && uv sync --group dev
export OPENROUTER_API_KEY=sk-or-...          # agent + judge both default to OpenRouter
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
  LiteLLM model string works (default `openrouter/deepseek/deepseek-v4-pro` at
  `--task-reasoning-effort xhigh`); `--max-turns` defaults to **200**, as LAB-AA.
- **Prompts** — `system_prompt` and `task_template` (`agent/prompts.py`), ported
  from AA's published LAB-AA prompts, adapted where they assume AA's sandbox.

Standard src layout, under `src/harvey_lab/`:

```
agent/       workspace.py (stage documents in, pull deliverables out)
             agent.py (wires code_exec + finish into Stirrup, runs one task)
             prompts.py
data/        dataset.py (task records + splits), fetch.py (on-demand download)
splits/      frozen {train,val,test}.txt task-id lists  → splits/README.md
evaluation/  scoring.py (batched judge, one task), run_eval.py (all tasks)
cli.py       `harvey-lab` console command; config.py: every model/budget knob
```

## How the work is graded

Each task ships a rubric of ~60 atomic PASS/FAIL criteria, each with a written
`match_criteria` standard and the deliverable(s) it applies to. A second LLM
acts as a **judge** (`evaluation/scoring.py`, default
`openrouter/deepseek/deepseek-v4-flash`): it reads only the deliverable(s) a
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

The train / val / test partition is **frozen**: the committed
`splits/{train,val,test}.txt` lists (1560 / 100 / 100 tasks) are the source of
truth — see [`src/harvey_lab/splits/README.md`](src/harvey_lab/splits/README.md).
`--split` picks one; `--limit N` runs the first N (any prefix stays
representative).

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory:

```bash
cd harvey_lab
uv sync --group dev
export OPENROUTER_API_KEY=sk-or-...   # covers both the agent and the judge
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

## Tests

```bash
uv run pytest -q
```

Hermetic — a scripted Stirrup client over the local shell backend, a stub judge,
and a fixture task tree: no network, no spend.

## Beaker prompt optimization

Beaker is configured to optimize the agent's two real LLM-visible prompts:
`system_prompt` and `task_template`. The spec constructs a fresh
`HarveyLabAgent` for each prompt candidate, so both optimized values are passed
to Stirrup's model call. The task model keeps its normal application defaults
for local runs; a Beaker-selected model is routed only inside the optimization
spec. The rubric judge remains fixed, which keeps scores comparable across
candidates.

`datasets/harvey-lab-pilot/` contains one real, pinned HARVEY-LABS train task
and one validation task. It is a wiring pilot, not enough data for a useful
optimization run. Its JSONL rows carry the task instructions, exact
deliverables, criterion rubric, and fingerprint from the source task; the
agent fetches or reads the matching task documents at rollout time.

To create a larger dataset from the same real source, first materialize the
chosen task folders and then export their labels:

```bash
export HARVEY_LAB_CACHE=/tmp/harvey-beaker-cache
uv run harvey-lab fetch --split train --limit 20 --cache-dir "$HARVEY_LAB_CACHE"
uv run harvey-lab fetch --split val --limit 10 --cache-dir "$HARVEY_LAB_CACHE"
uv run python scripts/export_beaker_dataset.py \
  --tasks-root "$HARVEY_LAB_CACHE/tasks" \
  --output-dir datasets/harvey-lab-v1 --train-limit 20 --val-limit 10
```

With `OPENROUTER_API_KEY` set, validate the real pilot locally (this calls the
task model and rubric judge, so it incurs provider usage):

```bash
export HARVEY_LAB_CACHE=/tmp/harvey-beaker-cache
beaker run dry-run --config '{"local_dataset_path":"datasets/harvey-lab-pilot"}'
```

For a hosted optimization, authenticate and create/select the specific target
agent before building the spec or uploading a larger dataset:

```bash
beaker login --agent --agent-name "Harvey LAB legal-agent prompts" --repo <owner/repo>
```

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
