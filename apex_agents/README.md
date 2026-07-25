# APEX-Agents

A self-contained analyst agent for Mercor's **APEX-Agents** benchmark (Law /
Investment Banking) — and how rilixai's GEPA loop optimizes its prompts. This
folder is one integrated demo: a realistic agent, its data, its grading, and
its rilixai integration, with no shared cookbook code.

## The agent

The agent plays a **domain analyst** (legal or investment-banking). For each
task it is given a "world" of source files and a question, and it must produce
a written deliverable by reading and reasoning over that world.

- **Input** — a per-case *world*: a set of source files (PDFs, spreadsheets,
  Word docs, text) plus the task prompt.
- **Tools** — a ReAct toolbelt combining meta tools (think / plan) with
  domain tools over the world files (list / open / read / search), terminating
  when the agent calls `final_answer` with its deliverable.
- **Loop** — a hand-rolled ReAct loop (reason → act → observe) over LiteLLM,
  bounded by `max_steps` and a `cost_limit` (USD). Long transcripts are
  compacted via a resummarization step (its own prompt).
- **Model** — a LiteLLM model string (default
  `openai/gpt-4.1-mini-2025-04-14`), so any provider LiteLLM routes to works.
- **Output** — the `final_answer` deliverable text.
- **Prompts** — three of them: `system_prompt`, `task_template`, and
  `resum_summary_prompt` (the transcript-compaction prompt). These are the
  strings rilixai optimizes.

The importable package lives under `src/apex_agents/` (standalone src-layout
project). Code map: `src/apex_agents/agent/agent.py` (the ReAct loop),
`src/apex_agents/agent/world/` (the world-files surface + tools),
`src/apex_agents/agent/prompts.py` (the seed prompts),
`src/apex_agents/config.py` (model / budget knobs). Paths below are relative to
`src/apex_agents/` unless noted.

### How the work is graded

Each task ships a rubric of atomic criteria. A second LLM acts as a **judge**:
it scores the agent's deliverable criterion-by-criterion and the scorer reports
`rubric_pass_rate` — the fraction of criteria passed
(`optimization/metrics.py`). That dense signal is what the recipe optimizes.

### The data (and its splits)

Cases come from the **private** HuggingFace dataset `mercor/apex-agents`
(requires an `HF_TOKEN` with access). The recipe splits by whole *world*, so
an evaluated candidate is scored for cross-world transfer rather than in-world
fit (`data/world_splits.py`). Which domain subset (`law` /
`investment_banking`) a run covers is chosen by which cases you export into the
uploaded dataset — not a per-trigger flag. Treat the scores as a
**before/after-optimization delta on this fixed harness**.

## How rilixai plugs in

Everything rilixai-specific lives in `optimization/`. The integration is one
`Spec` binding four things (`optimization/spec.py`):

| Spec field | What it is here |
|---|---|
| `seed_targets` | the three seed prompts (`agent/prompts.py`) |
| `data_loader` | `ApexAgentsDataLoader` — maps one uploaded JSONL row → one `Case` (`data/dataset.py`) |
| `run_case` | async adapter that runs the agent on a case, grades it, returns a `CaseResult` (`optimization/runtime.py`) |
| `scorer` | `ApexAgentsScorer` — reads `rubric_pass_rate` back off the judged result (`optimization/metrics.py`) |

The optimizer drives `run_case(case=..., targets=..., runtime=...)` with each
candidate prompt bundle; GEPA keeps the prompts that score higher.

**Optional model selection.** `run_case` respects `runtime.model` when the
optimizer selects a model for a rollout (overriding `task_model`), and
otherwise uses the recipe's own default. Ordinary runs need no model from
rilixai.

**Recipe knobs travel under `extra`.** The strict launch config only accepts a
fixed set of top-level keys (`max_metric_calls`, …); this recipe's own knobs
(`task_model`, `task_temperature`, `judge_model`, `max_steps`, `cost_limit`)
travel under the launch config's `extra` block, which `build_spec` reads via
`ctx.config["extra"]` (see `_DEFAULT_SANDBOX_CONFIG` at the top of
`optimization/spec.py`).

**Local vs hosted.** This recipe depends only on the lightweight `rilixai`
SDK. The local CLI (`cli.py`) covers the two SDK-only paths — `validate` and
`evaluate` (`optimization/local_eval.py` scores one candidate the way the
hosted runner does). The full GEPA optimize loop runs server-side via
`rilixai run`; `sandbox.py` builds + promotes + triggers it.

## Install

This recipe is a standalone uv project — set it up from its own folder:

```bash
cd apex_agents
uv sync --group dev
```

Env vars (needed for any run that calls a model):

```bash
export HF_TOKEN=hf_...          # private dataset access
export OPENAI_API_KEY=sk-...    # agent (gpt-4.1-mini default)
export GOOGLE_API_KEY=...       # judge (gemini-2.5-flash default)
```

## Run locally

```bash
# Validate the spec structure offline (no network, no dataset download)
uv run python -m apex_agents.cli validate --domain law

# Evaluate one candidate (omit --candidate-json to score the seed prompts)
uv run python -m apex_agents.cli evaluate \
    --domain law --candidate-json path/to/candidate.json
```

`--val-worlds` holds out whole worlds when `--split validation` builds the
fixed val pool, so an evaluated candidate is scored for cross-world transfer
rather than in-world fit. See `--help` for all flags.

## Run on Modal (rilixai sandbox)

`optimization/spec.py` registers a `@spec(name="apex-agents")` factory that
rilixai's sandbox runs. `sandbox.py` builds the image, promotes it to
`apex-agents@production`, and triggers a run in one shot.

**A dataset upload is required.** The spec sources its cases from an uploaded
JSONL dataset via `ApexAgentsDataLoader` (see
`ApexAgentsDataLoader.dataset_schema` in `data/dataset.py`). A run triggered
with no dataset reference is rejected at startup. Which domain subset (`law` /
`investment_banking`) a run optimizes over is decided by which cases you export
into the uploaded dataset — not a per-trigger flag. Upload once, then trigger:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/
export RILIXAI_AGENT_KEY=apex-agents   # agent the trigger targets (or pass --agent)

# One-time (or when the data changes): upload the JSONL split as a dataset.
uv run rilixai dataset upload --name apex-agents-dataset path/to/jsonl-dir/

uv run sandbox.py --build   # build + promote + trigger
uv run sandbox.py           # trigger only (current @production)
```

The trigger defaults to `--dataset apex-agents-dataset@production` and
`--spec apex-agents@production`; override either to pin a specific revision.
Provider keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) and `HF_TOKEN` are bound as
project-level secrets on rilixai's side, injected into each sandbox. The domain
subset and train/val split come from the uploaded dataset, so there are no
`domain`/`train_size`/`val_size`/`val_worlds` knobs. Roll back with
`uv run rilixai spec promote apex-agents v<older-sha>`.

CI (`.github/workflows/push-spec.yml`) runs `sandbox.py --build --no-trigger`
on every merge to `main` that touches `apex_agents/`: it ships the image and
flips `@production` without spending LLM tokens on a smoke run.

## Tests

```bash
uv run python -m pytest -q
```

Hermetic — a `FakeWorld` shim + stub judge, no network.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- `max_steps=60` / `cost_limit=$3` are demo-bounded; raise for parity.
