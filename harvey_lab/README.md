# Harvey LAB

A self-contained legal knowledge-worker agent — and how rilixai's GEPA loop
optimizes its prompts. This folder is one integrated demo: a realistic agent,
its data, its grading, and its rilixai integration, with no shared cookbook
code.

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
  runs on). This recipe supplies only the *domain*: the workspace, the tools,
  and the two prompts.
- **Model** — a LiteLLM model string (default `openai/gpt-4.1-mini`), so any
  provider LiteLLM routes to works. `max_turns` caps the loop.
- **Prompts** — two of them, `system_prompt` and `task_template`. These are the
  strings rilixai optimizes.

The importable package lives under `src/harvey_lab/` (standalone src-layout
project). Code map: `src/harvey_lab/agent/workspace.py` (the workspace + file
tools), `src/harvey_lab/agent/agent.py` (wires the tools into Stirrup and runs a
task), `src/harvey_lab/agent/prompts.py` (the seed prompts),
`src/harvey_lab/config.py` (model / budget knobs). Paths below are relative to
`src/harvey_lab/` unless noted.

### How the work is graded

Each task ships a rubric of ~60 atomic PASS/FAIL criteria, each with a written
`match_criteria` standard and the deliverable(s) it applies to. A second LLM
acts as a **judge**: for each criterion it reads only the deliverable(s) that
criterion names (deliverable-scoped) and returns PASS/FAIL. Two numbers come
out (`optimization/scoring.py`):

- `all_pass` (0/1) — the LAB headline metric: `1.0` **iff every** criterion
  passes, else `0.0`.
- `criterion_pass_rate` — the fraction of criteria passed.

`all_pass` is the metric of record but an extremely sparse training signal (one
missed criterion zeroes the task), so the recipe **optimizes the dense
`criterion_pass_rate`** and reports `all_pass` alongside. To optimize `all_pass`
directly, pass `field_weights={"all_pass": 1.0}` when building the spec.

### The data (and what it is *not*)

Cases come from the **public** GitHub repo
[`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs), pinned to a
commit in `config.py`. Documents are fetched at run time from that commit (only
their filenames are stored in the dataset), so a spec version stays
reproducible as the upstream benchmark grows. `data/task_splits.py` splits by
whole practice area, so train / held-out legal domains stay disjoint.

This is **not** the LAB-AA leaderboard setup. The leaderboard runs on Harvey's
**private** 120-task set through AA's exact Stirrup config; this recipe runs a
slimmed, text-only agent on the **public** tasks with a cheaper judge and a
bounded turn budget. The public data may be contamination-exposed. Treat the
scores as a **before/after-optimization delta on this fixed harness**, not a
leaderboard-comparable number.

## How rilixai plugs in

Everything rilixai-specific lives in `optimization/`. The integration is one
`Spec` binding four things (`optimization/spec.py`):

| Spec field | What it is here |
|---|---|
| `seed_targets` | the two seed prompts (`agent/prompts.py`) |
| `data_loader` | `HarveyLabDataLoader` — maps one uploaded JSONL row → one `Case` (`data/dataset.py`) |
| `run_case` | async adapter that runs the agent on a case, grades it, returns a `CaseResult` (`optimization/runtime.py`) |
| `scorer` | `HarveyLabScorer` — reads the precomputed `all_pass` / `criterion_pass_rate` back off the result (`optimization/scoring.py`) |

The optimizer drives `run_case(case=..., targets=..., runtime=...)` with each
candidate prompt bundle; GEPA keeps the prompts that score higher.

**Optional model selection.** `run_case` respects `runtime.model` (and
`runtime.provider`) when the optimizer selects a model for a rollout — e.g. a
multi-model benchmark — and otherwise uses the recipe's own production model.
Ordinary runs need no model from rilixai. See `_selected_model` in
`optimization/runtime.py`.

**Local vs hosted.** This recipe depends only on the lightweight `rilixai` SDK.
The local CLI (`cli.py`) covers the two SDK-only paths — `validate` and
`evaluate` (`optimization/local_eval.py` scores one candidate the way the hosted
runner does). The full GEPA optimize loop runs server-side via `rilixai run`;
`sandbox.py` builds + promotes + triggers it.

## Install

This recipe is a standalone uv project — set it up from its own folder:

```bash
cd harvey_lab
uv sync --group dev
```

Env vars (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # agent (gpt-4.1-mini default)
export GOOGLE_API_KEY=...       # judge (gemini-3.5-flash default)
```

No dataset token is required — the LAB repo is public.

## Run locally

```bash
# Validate the spec structure offline (no network, no documents, no LLM)
uv run python -m harvey_lab.cli validate

# Evaluate one candidate against a local checkout of the benchmark
git clone https://github.com/harveyai/harvey-labs
uv run python -m harvey_lab.cli evaluate \
    --tasks-root harvey-labs/tasks \
    --practice-areas contracts,corporate-ma \
    --test-size 10 \
    --candidate-json path/to/candidate.json    # omit to score the seed
```

See `--help` for all flags.

## Run on Modal (rilixai sandbox)

`optimization/spec.py` registers a `@spec(name="harvey-lab")` factory that
rilixai's sandbox runs. `sandbox.py` builds the image, promotes it to
`harvey-lab@production`, and triggers a run in one shot.

**A dataset upload is required.** The spec sources its cases from an uploaded
JSONL dataset via `HarveyLabDataLoader` (see
`HarveyLabDataLoader.dataset_schema` in `data/dataset.py`); a run triggered with
no dataset reference is rejected at startup. Export the split from a local
checkout, then upload:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/
export RILIXAI_AGENT_KEY=harvey-lab   # agent the trigger targets (or pass --agent)

git clone https://github.com/harveyai/harvey-labs
# --test-areas carves a disjoint test.jsonl the optimizer scores the winner on
# post-optimization (unbiased held-out); omit it for train/val only.
# --embed-documents base64-bundles each task's documents into the rows so the
# run reads them from the artifact (no run-time fetch, no rate-limit).
uv run python scripts/export_harvey_lab_dataset.py --tasks-root harvey-labs/tasks \
  --val-areas 3 --test-areas 3 --embed-documents
uv run rilixai dataset upload --name harvey-lab-dataset \
  --split train=883 --split val=123 --split test=126 scripts/_datasets/harvey_lab

uv run sandbox.py --build   # build + promote + trigger
uv run sandbox.py           # trigger only (current @production)
```

The trigger defaults to `--dataset harvey-lab-dataset@production` and
`--spec harvey-lab@production`. `--max-metric-calls` is the primary cost knob
(default 50, a smoke budget) and is the only optimizer setting passed at the top
level; the reflection model / minibatch / seed are chosen by the server-side
recipe preset. The recipe's own knobs (task model, judge model, turn budget)
travel under the launch config's `extra` block — see `trigger_run` in
`sandbox.py` and how `build_spec` reads `ctx.config["extra"]`. Provider keys are
bound as project-level secrets on rilixai's side and injected into each sandbox.
Roll back with `uv run rilixai spec promote harvey-lab v<older-sha>`.

## Tests

```bash
uv run python -m pytest -q
```

Hermetic — a fixture task tree + a scripted Stirrup client + a stub judge, no
network.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- `max_turns=40` is demo-bounded; raise for parity with a full LAB-AA harness.
- The text-only toolbelt can't produce real `.docx`/`.xlsx` redlines, so some
  format-specific LAB criteria are unwinnable here by construction.
