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

## Sanity-check locally

`rilixai dry-run` builds the spec and runs one case with the seed candidate
— no push, no optimizer loop. Run it from the member directory so it picks up
`[tool.rilixai.spec]`:

```bash
cd apex_agents
uv run rilixai dry-run --config '{"domain": "law", "train_size": 1, "val_size": 1}'
```

It prints the agent output, the per-field `rubric_pass_rate`, and the feedback
narratives the reflection LM would see.

## Run on rilixai

`rilixai_spec.py` registers the `@spec(name="apex-agents")` runner. The full
build → optimize → pull flow:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/

# Build the image + promote it to apex-agents@production (run from repo root)
SPEC_VERSION=v$(git rev-parse --short HEAD)
uv run rilixai push --member apex_agents --version "$SPEC_VERSION"

# Queue a run; --val-worlds holds out whole worlds so GEPA selects for
# cross-world transfer, not in-world fit
cd apex_agents
uv run rilixai trigger --config '{"domain": "law", "train_size": 25, "val_size": 20, "val_worlds": 2, "max_metric_calls": 100}'

# Watch it, then download the optimized candidate + per-split reports
uv run rilixai status <run_id>
uv run rilixai pull <run_id> --output-dir apex_agents_results
```

The run config keys are the fields of `ApexAgentsSandboxConfig` in
`rilixai_spec.py`. Provider keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) and
`HF_TOKEN` are bound as project-level secrets on rilixai's side, injected into
each sandbox. Roll back with
`uv run rilixai spec promote apex-agents --version <older-sha>`.

CI (`.github/workflows/push-spec.yml`) runs
`rilixai push --member apex_agents --version v<short-sha>` on every merge to
`main` that touches `apex_agents/`: it ships the image and flips `@production`
without spending LLM tokens on a smoke run.

## Tests

```bash
uv run python -m pytest apex_agents/tests -q
```

Hermetic — a `FakeWorld` shim + stub judge, no network.

## Adapting this recipe to your agent

The whole rilixai integration is `rilixai_spec.py` — one `@spec`-decorated
`ApexAgentsRunner(BaseCaseRunner)` (with an LLM rubric judge as a custom
comparator). To wire your own agent, see the cookbook's
**[`ONBOARDING.md`](../ONBOARDING.md)** for the full contract.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- `max_steps=60` / `cost_limit=$3` are demo-bounded; raise for parity.
