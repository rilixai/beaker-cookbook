# rilixai-cookbook
<div align="center">
  <img width="227" height="225" alt="31fdec74-bda3-4813-99ab-76f59f1f4484" src="https://github.com/user-attachments/assets/174d316a-24c0-4832-9f93-9c0a77a19433" />

</div>


*A collection of [rilixai](https://github.com/rilixai/rilixai) recipes
that demonstrate rilixai's continual learning capabilities across different agent shapes, tasks, and
production setups.*



## Recipes

```
harvey_lab/       Legal deliverables — Stirrup-harnessed agent on Harvey's public LAB tasks
hotpotqa/         Multi-hop QA — PydanticAI tool-using agent (retrieve_k + summarize)
apex_agents/      Law / IB analyst tasks — ReAct toolbelt agent on Mercor's APEX-Agents
```

Each folder is a **self-contained integrated demo**: a complete agent
plus its rilixai integration, with nothing shared across recipes. A
folder contains everything you need to reproduce the recipe — a README
that explains the agent first and then how rilixai plugs in, a local
CLI (`cli.py`) for fast iteration on your laptop, an optional Modal
sandbox path (`sandbox.py`) for hosted runs at scale, and a hermetic
test suite so you can verify the harness end-to-end before spending
anything on LLM calls.

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Each
recipe is its own standalone uv project (its own `pyproject.toml` and
`uv.lock`), so you set up and run one recipe entirely from its folder —
there is no root workspace to install:

```bash
cd hotpotqa            # or harvey_lab / apex_agents
uv sync --group dev
```

From there follow the commands in that recipe's README.

## Configuration

The recipes need credentials for the LLM provider and, if you're
using the hosted path, for the rilixai control plane. The easiest
setup is a `.env` file at the cookbook root with the variables below;
they'll be picked up automatically by both the local CLIs and the
sandbox scripts.

| Variable | Required for | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Local runs (every recipe's `cli.py`) | [OpenAI dashboard](https://platform.openai.com/api-keys) |
| `RILIXAI_API_KEY` | Hosted runs (every recipe's `sandbox.py`) | rilixai dashboard |
| `RILIXAI_API_BASE_URL` | Hosted runs | rilixai dashboard — looks like `https://<id>.execute-api.us-east-2.amazonaws.com/prod/` |

For hosted runs you don't need `OPENAI_API_KEY` on your machine — it's
bound at the rilixai *project* level, and each sandbox container that
spawns inherits it as an env var. Set it once in the rilixai dashboard
and forget about it locally.

Per-recipe knobs like model choice, dataset size, and retrieval mode
are documented in each recipe's README and surfaced as CLI flags, so
you can usually go from "first time looking at this folder" to "smoke
run" without touching code.

## What runs where

Each recipe ships two entry points so you can pick the right tool for
the job:

| Path | Where it runs | When to use |
|---|---|---|
| `<recipe>/cli.py` | Your laptop | Fast iteration, local debugging, no rilixai account needed |
| `<recipe>/sandbox.py` | rilixai hosted Modal sandbox | Real optimization runs at scale, scheduled retraining, sharing optimized prompts across teams |

The local CLI uses your `OPENAI_API_KEY` directly and writes results
to local files. The hosted path packages the recipe into a Modal
image, queues the run through rilixai's API, and writes the optimized
prompts back to your rilixai project so any service of yours can
fetch them at runtime.

## Development

Run the checks from within a recipe folder (each recipe is standalone):

```bash
cd hotpotqa            # or harvey_lab / apex_agents
uv sync --group dev
uv run python -m pytest -q
uv run ruff check
uv run ruff format --check
uv run python -m mypy
```

## rilixai dependency

Pinned via `git+ssh` at a specific main-branch SHA until a fresh
rilixai is published to PyPI. See each recipe's `pyproject.toml`
for the current pin.
