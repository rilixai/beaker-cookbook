# APEX-Agents

Mercor's **APEX-Agents** benchmark (Law / Investment Banking) as a ReAct
toolbelt agent, optimized by rilixai's GEPA loop.

- **Agent** — a ReAct loop over per-case "world" files (PDFs, spreadsheets,
  docs) with meta + domain tools; terminates by calling `final_answer`.
- **Optimized prompts** — `system_prompt`, `task_template`,
  `resum_summary_prompt`.
- **Score** — `rubric_pass_rate` from an LLM judge against each task's rubric.

Cases come from the private HF dataset `mercor/apex-agents`.

## Install

```bash
uv sync --all-packages --group dev
```

Env vars (needed for any run that calls a model):

```bash
export HF_TOKEN=hf_...          # private dataset access
export OPENAI_API_KEY=sk-...    # agent (gpt-4.1-mini default)
export GOOGLE_API_KEY=...       # judge (gemini-2.5-flash default)
```

## Run locally

This recipe depends on the lightweight `rilixai` SDK only. The local CLI
covers the two SDK-only paths — `validate` (offline structural check) and
`evaluate` (score one candidate via the SDK `run_case` + scorer loop). The
full GEPA optimize loop runs server-side via `rilixai run` (see the Modal
section below); the optimizer engine lives in the optional `rilixai-runtime`
package, not in this recipe.

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

**A dataset upload is required.** The migrated spec no longer loads data
itself — the optimizer reads its cases from an uploaded JSONL dataset via
`ApexAgentsDataLoader` (see `ApexAgentsDataLoader.dataset_schema` in
`apex_agents/data/dataset.py`). A run triggered with no dataset reference is
rejected at startup. Which domain subset (`law` / `investment_banking`) a run
optimizes over is decided by which cases you export into the uploaded dataset —
not a per-trigger flag. Upload once, then trigger:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/
export RILIXAI_AGENT_KEY=apex-agents   # agent the trigger targets (or pass --agent)

# One-time (or when the data changes): upload the JSONL split as a dataset.
uv run rilixai dataset upload --name apex-agents-dataset path/to/jsonl-dir/

uv run apex_agents/sandbox.py --build   # build + promote + trigger
uv run apex_agents/sandbox.py           # trigger only (current @production)
```

The trigger defaults to `--dataset apex-agents-dataset@production` and
`--spec apex-agents@production`; override either to pin a specific revision.

Provider keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) and `HF_TOKEN` are bound
as project-level secrets on rilixai's side, injected into each sandbox.

To trigger from code, or to tune the agent knobs (`max_metric_calls`, models,
…), call `client.create_optimization_run(...)` with a `dataset_ref` — the run
config keys are documented in `_DEFAULT_SANDBOX_CONFIG` at the top of
`apex_agents/optimization/spec.py`. The domain subset and train/val split come
from the uploaded dataset, so there are no `domain`/`train_size`/`val_size`/
`val_worlds` knobs. Roll back with
`uv run rilixai spec promote apex-agents v<older-sha>`.

## Tests

```bash
uv run python -m pytest apex_agents/tests -q
```

Hermetic — a `FakeWorld` shim + stub judge, no network.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- `max_steps=60` / `cost_limit=$3` are demo-bounded; raise for parity.
