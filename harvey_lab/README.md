# Harvey LAB

A self-contained **legal knowledge-worker agent** and a local rubric evaluation
harness for it. The agent reads a case folder, produces the requested written
deliverables, and is graded criterion-by-criterion against Harvey's public
[Legal Agent Benchmark (LAB)](https://www.harvey.ai/blog/introducing-harveys-legal-agent-benchmark).

## The agent

The agent plays a **junior lawyer**. For each task it is handed a case folder
and a to-do list, and it must produce the requested written work.

- **Input** — a per-task workspace: a read-only `documents/` tree (the source
  record: contracts, emails, spreadsheets, PDFs) plus the task instructions and
  a list of deliverables to produce.
- **Tools** — a small file toolbelt over that workspace: `list_files`,
  `read_document` (extracts text from `.docx` / `.xlsx` / `.pdf` / `.eml` /
  plain text), `grep_documents`, `write_deliverable`, `edit_deliverable`, and
  `finish`. Deliverables are written as text into a writable `output/` dir.
- **Loop** — the tool-use loop, context management, and message plumbing come
  from [Stirrup](https://github.com/ArtificialAnalysis/Stirrup), Artificial
  Analysis' agent harness (the one their
  [Harvey LAB-AA leaderboard](https://artificialanalysis.ai/evaluations/harvey-lab-aa)
  runs on). This package supplies only the *domain*: the workspace, the tools,
  and the two prompts.
- **Model** — a LiteLLM model string (default `openai/gpt-4.1-mini`), so any
  provider LiteLLM routes to works. `max_turns` caps the loop.
- **Prompts** — two of them, `system_prompt` and `task_template`
  (`agent/prompts.py`).

Code map:

| Path | What it holds |
|---|---|
| `agent/workspace.py` | the per-task workspace + the file tools |
| `agent/agent.py` | wires the tools into Stirrup and runs one task |
| `agent/prompts.py` | the two prompts |
| `data/dataset.py` | loads LAB task records from a local checkout + reads the splits |
| `splits/{train,val,test}.txt` | frozen task-id lists (see `splits/README.md`) |
| `evaluation/scoring.py` | grades ONE task: the batched judge + all-pass aggregation |
| `evaluation/run_eval.py` | runs the agent + scoring across the dataset (bounded concurrency) |
| `evaluation/utils.py` | JSON + summary serialization for the CLI |
| `config.py` | model / budget / timeout knobs |
| `cli.py` | run the agent, or run + grade it |

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
and [Applied Compute](https://www.appliedcompute.com/case-studies/harvey), both
cited in the code). The default judge is `deepseek/deepseek-v4-flash`. Two
numbers come out per task:

- `all_pass` (0/1) — the LAB headline metric: `1.0` **iff every** criterion
  passes, else `0.0`.
- `criterion_pass_rate` — the fraction of criteria passed (a denser view of the
  same grading, since a single missed criterion zeroes `all_pass`).

The dataset runner (`evaluation/run_eval.py`) averages both across tasks as
`all_pass_rate` / `criterion_pass_rate`. A task whose rubric has no scoreable
criteria is **unscoreable** and excluded from the averages; a task that errors
counts as `0` (a real failure must deflate, not inflate, the metrics).

## The data (and what it is *not*)

Tasks come from the **public** GitHub repo
[`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs). Clone it at
the pinned `config.HARVEY_LABS_COMMIT` and point `--tasks-root` at its `tasks/`
directory. Task directories are discovered recursively (larger areas nest
sub-categories, e.g. `contracts/banking/<slug>`), so a task ID is the directory
path under `tasks/`. The train / val / test partition is **frozen**: the
committed `splits/{train,val,test}.txt` lists (1560 / 100 / 100 tasks, drawn to
follow the benchmark's natural practice-area distribution) are the source of
truth — see `splits/README.md`. `--split` picks one; `--limit N` runs just the
first N (the lists are ordered so any prefix stays representative).

This is **not** the LAB-AA leaderboard setup. The leaderboard runs on Harvey's
**private** 120-task set through AA's exact Stirrup config; this repo runs a
slimmed, text-only agent on the **public** tasks with a cheaper judge and a
bounded turn budget. The public data may be contamination-exposed, so treat the
scores as a self-contained measurement on this fixed harness, **not** a
leaderboard-comparable number.

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory:

```bash
cd harvey_lab
uv sync --group dev
```

Provider keys (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # agent (gpt-4.1-mini default)
export DEEPSEEK_API_KEY=...     # judge (deepseek-v4-flash default)
```

No dataset token is required — the LAB repo is public.

## Run

```bash
git clone https://github.com/harveyai/harvey-labs
git -C harvey-labs checkout 1da4750171bc5a534960b3d82d15ba7fd2cf653f

# Run the agent and dump its deliverables (no grading), on 10 test tasks:
uv run python -m harvey_lab.cli run \
    --tasks-root harvey-labs/tasks \
    --split test --limit 10 \
    --output-dir harvey_lab_run

# Run the agent AND grade every rubric criterion:
uv run python -m harvey_lab.cli evaluate \
    --tasks-root harvey-labs/tasks \
    --split test --limit 10 \
    --output-dir harvey_lab_run
```

`evaluate` writes `eval_summary.json` (aggregate `all_pass_rate` /
`criterion_pass_rate` + case counts) and `eval_outputs.json` (per-task results)
to `--output-dir`. See `--help` for all flags (`--split {train,val,test}`,
`--limit`, `--max-concurrency`, `--task-model`, `--judge-model`,
`--judge-batch-size`, …).

## Tests

```bash
uv run python -m pytest -q
```

Hermetic — a fixture task tree + a scripted Stirrup client + a stub judge, no
network.

## Notes

- `max_turns=40` is a smoke-run budget; raise it for parity with a full LAB-AA
  harness.
- The text-only toolbelt can't produce real `.docx`/`.xlsx` redlines, so some
  format-specific LAB criteria are unwinnable here by construction.
