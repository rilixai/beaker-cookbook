# rilixai-cookbook
<div align="center">
  <img width="227" height="225" alt="31fdec74-bda3-4813-99ab-76f59f1f4484" src="https://github.com/user-attachments/assets/174d316a-24c0-4832-9f93-9c0a77a19433" />

</div>


*A collection of standalone agent recipes — each one a runnable agent
plus a local evaluation — across different agent shapes, tasks, and
production setups.*



## Recipes

```
harvey_lab/       Legal-research agent over the Harvey/LegalBench-style corpus
hotpotqa/         Multi-hop QA — PydanticAI tool-using agent (retrieve_k + summarize)
apex_agents/      Professional knowledge-work tasks — ReAct toolbelt agent, LLM rubric judge
```

Each folder contains everything you need to reproduce the recipe: a
README with the canonical reproduce commands and expected scores, a
CLI (`cli.py`) exposing `run` and `evaluate`, and a hermetic test
suite so you can verify the harness end-to-end before spending
anything on LLM calls.

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Each recipe is
a self-contained, standalone uv project (its own `pyproject.toml` +
`uv.lock`) — there is no root workspace. Pick a recipe, install it, and
follow the commands in its README:

```bash
cd harvey_lab        # or hotpotqa / apex_agents
uv sync --group dev
```

## Configuration

The recipes need credentials for the LLM providers they call. The
easiest setup is a `.env` file at the cookbook root; it is picked up
automatically by the CLIs.

| Variable | Required for | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Every recipe's `cli.py` | [OpenAI dashboard](https://platform.openai.com/api-keys) |

Some recipes need extra credentials — a judge model's provider key, or
a HuggingFace token for a gated dataset. Each recipe's README lists
what it needs.

Per-recipe knobs like model choice, dataset size, and retrieval mode
are documented in each recipe's README and surfaced as CLI flags, so
you can usually go from "first time looking at this folder" to "smoke
run" without touching code.

## What runs where

Everything runs locally. `<recipe>/cli.py` exposes two commands:
`run` executes the agent over the selected cases and dumps its
outputs, `evaluate` also scores them and writes a summary. Both read
your provider keys from the environment and write results to a local
`--output-dir`.

## Development

Run the checks from inside a recipe directory — each recipe carries its own
lint/type/test config:

```bash
cd harvey_lab        # or hotpotqa / apex_agents
uv sync --group dev
uv run python -m pytest -q
uv run ruff check
uv run ruff format --check
uv run python -m mypy
```

