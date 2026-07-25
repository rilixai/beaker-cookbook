# APEX-Agents

A self-contained **professional knowledge-worker agent** and a local rubric
evaluation harness for it. The agent explores a per-task "world" of business
documents, answers the task, and is graded criterion-by-criterion against
Mercor's [APEX-Agents](https://huggingface.co/datasets/mercor/apex-agents)
benchmark (Law / Investment Banking).

## The agent

For each task the agent is handed a **world** — the deal room / matter folder
the task lives in — and a written request, and it must produce the answer.

- **Input** — a read-only per-task world: a directory tree of spreadsheets,
  PDFs, Word documents and text files, plus the task's own input files.
- **Tools** — the toolbelt starts *empty*. Meta tools
  (`toolbelt_list_tools`, `toolbelt_inspect_tool`, `toolbelt_add_tool`,
  `toolbelt_remove_tool`, `todo_write`, `final_answer`) let the agent discover
  and mount the domain tools it needs: `list_files`, `read_file`,
  `read_spreadsheet` (per-sheet, `.xlsx` + legacy `.xls`), `read_pdf`,
  `read_docx`, and `search_files`.
- **Loop** — a native async ReAct loop: think → call tools → observe, until
  `final_answer` is called. Answering is gated on the agent's own todo list
  being closed, so it can't punt mid-plan. When the conversation outgrows the
  context budget it is compacted (**ReSum**) and the loop continues.
- **Model** — a LiteLLM model string (default `openai/gpt-4.1-mini`), so any
  provider LiteLLM routes to works. `max_steps` and `cost_limit` bound the loop.
- **Prompts** — three of them: `system_prompt`, `task_template`, and
  `resum_summary_prompt` (`agent/prompts.py`).

Code map:

| Path | What it holds |
|---|---|
| `agent/world/world.py` | the per-task world (zip extraction + file readers) |
| `agent/agent.py` | the ReAct loop, the toolbelt, and the LiteLLM wrapper |
| `agent/prompts.py` | the three prompts |
| `data/dataset.py` | loads task records + world metadata from HuggingFace |
| `data/world_splits.py` | deterministic world-level splitters |
| `evaluation/scoring.py` | the per-criterion LLM judge + `rubric_pass_rate` |
| `evaluation/local_eval.py` | the bounded-concurrency batch evaluator |
| `evaluation/report.py` | the JSON artifacts |
| `config.py` | model / budget / timeout knobs |
| `cli.py` | run the agent, or run + grade it |

## How the work is graded

Each task ships a rubric of atomic criteria written in plain English. A second
LLM acts as a **judge**: for each criterion it reads the task and the agent's
final answer and returns `VERDICT: MET` / `VERDICT: NOT MET` — an ambiguous or
failed judgement is conservatively NOT MET. The per-task score
(`rubric_pass_rate`) is the fraction of criteria met.

The batch evaluator (`evaluation/local_eval.py`) averages that across tasks. A
task whose rubric has no scoreable criteria is **unscoreable** and excluded
from the average; a task that errors counts as `0` (a real failure must
deflate, not inflate, the metric).

`data/world_splits.py` carves the validation pool by *whole world*, so
`evaluate --split all` also reports a `rubric_pass_rate_heldout` over the
worlds outside that pool.

## The data

Tasks come from the **gated** HuggingFace dataset
[`mercor/apex-agents`](https://huggingface.co/datasets/mercor/apex-agents):
`tasks_and_rubrics.json`, `world_descriptions.json`, and one zip per world
holding its files. Request access, then authenticate (below). By default only
the `Investment Banking` subset is loaded (`--domain`, `--domain ""` for all).

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory:

```bash
cd apex_agents
uv sync --group dev
```

Credentials:

```bash
export HF_TOKEN=hf_...          # gated dataset access
export OPENAI_API_KEY=sk-...    # agent (gpt-4.1-mini default)
export GOOGLE_API_KEY=...       # judge (gemini-3.5-flash default)
```

## Run

```bash
# Run the agent and dump its answers (no grading):
uv run python -m apex_agents.cli run \
    --test-size 10 \
    --output-dir apex_agents_run

# Run the agent AND grade every rubric criterion:
uv run python -m apex_agents.cli evaluate \
    --test-size 10 \
    --output-dir apex_agents_run
```

`run` writes `run_outputs.json` (per-task answer + loop telemetry, or an
`error` entry for a task that failed — one failure never aborts the batch).
`evaluate` writes `eval_summary.json` (aggregate `rubric_pass_rate` + case
counts) and `eval_outputs.json` (per-task results). See `--help` for all flags
(`--split`, `--val-worlds`, `--val-size`, `--max-concurrency`, `--task-model`,
`--judge-model`, `--max-steps`, `--cost-limit`, `--cache-dir`, …).

`--no-network` refuses the dataset download, the world factory, and the judge —
a dry-run guard against accidental spend.

## Tests

```bash
uv run python -m pytest -q
```

Hermetic — a `FakeWorld` shim + a scripted model + a stub judge, no network.

## Notes

- `max_steps=60` / `cost_limit=$3` are demo-bounded; raise them for parity with
  the reference harness (which allows 250 steps).
- The judge is an LLM, so scores carry judge noise; keep the judge model fixed
  when comparing runs.
