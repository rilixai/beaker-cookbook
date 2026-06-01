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

```bash
# Optimize on a small Law slice
uv run python -m apex_agents.cli optimize \
    --domain law --train-size 25 --val-size 20 --val-worlds 2 \
    --max-metric-calls 100 --output-dir apex_agents_results/smoke

# Evaluate (omit --candidate-json to score the seed prompts)
uv run python -m apex_agents.cli evaluate \
    --domain law --candidate-json apex_agents_results/smoke/best_candidate.json
```

`--val-worlds` holds out whole worlds for validation so GEPA selects for
cross-world transfer, not in-world fit. See `--help` for all flags.

## Run on Modal (rilixai sandbox)

`optimization/spec.py` registers a `@spec(name="apex-agents")` factory that
rilixai's sandbox runs. `sandbox.py` builds the image, promotes it to
`apex-agents@production`, and triggers a run in one shot:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/

uv run apex_agents/sandbox.py --build   # build + promote + trigger
uv run apex_agents/sandbox.py           # trigger only (current @production)
```

Provider keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) and `HF_TOKEN` are bound
as project-level secrets on rilixai's side, injected into each sandbox.

To trigger from code, or to tune knobs (`max_metric_calls`, `domain`,
`train_size`, models, …), call `client.create_optimization_run(...)` — the
run config keys are documented in `_DEFAULT_SANDBOX_CONFIG` at the top of
`apex_agents/optimization/spec.py`. Roll back with
`uv run rilixai spec promote apex-agents --version <older-sha>`.

## Tests

```bash
uv run python -m pytest apex_agents/tests -q
```

Hermetic — a `FakeWorld` shim + stub judge, no network.

## Adapting this recipe to your agent

The whole rilixai integration is `rilixai_spec.py` — one `@spec`-decorated
`ApexAgentsRunner(BaseSampleRunner)` (with an LLM rubric judge as a custom
comparator). To wire your own agent, see the cookbook's
**[`ONBOARDING.md`](../ONBOARDING.md)** for the full contract.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- `max_steps=60` / `cost_limit=$3` are demo-bounded; raise for parity.
