# rilixai-cookbook
<div align="center">
  <img width="227" height="225" alt="31fdec74-bda3-4813-99ab-76f59f1f4484" src="https://github.com/user-attachments/assets/174d316a-24c0-4832-9f93-9c0a77a19433" />

</div>


*A collection of [rilixai](https://github.com/rilixai/rilixai) recipes
that demonstrate rilixai's continual learning capabilities across different agent shapes, tasks, and
production setups.*



## Recipes

```
hotpotqa/         Multi-hop QA — PydanticAI tool-using agent (retrieve_k + summarize)
apex_agents/      Mercor APEX-Agents (Law / IB) — ReAct toolbelt agent + LLM rubric judge
```

Each folder is the whole rilixai integration in one file — a single
`@spec`-decorated `BaseCaseRunner` in `rilixai_spec.py` — plus a README
with the canonical commands and a hermetic test suite so you can verify
the harness end-to-end before spending anything on LLM calls. There's no
per-recipe CLI or build script: the top-level `rilixai` CLI (`dry-run`,
`push`, `trigger`, `status`, `pull`) drives every recipe.

## Quick start

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). The
cookbook is a uv workspace, so a single command installs every recipe
together:

```bash
uv sync --all-packages --group dev
```

From there each recipe is independently runnable — pick one and
follow the commands in its README.

## Onboarding your own agent

The recipes here are worked examples of one contract: a single
`@spec`-decorated `BaseCaseRunner` class wires your production agent
into rilixai's optimizer. **[`ONBOARDING.md`](ONBOARDING.md)** is the
full contract reference — what rilixai expects from your agent, the
`ComponentApplier` framework cheatsheet (PydanticAI, OpenAI, Anthropic,
LangChain, …), data-loading patterns, the `FieldConfig` scoring
cookbook, and the path from `rilixai init spec` to a queued run.

## Configuration

The recipes need credentials for the LLM provider and, for the hosted
path, for the rilixai control plane. The easiest setup is a `.env` file
at the cookbook root with the variables below; they're picked up
automatically by the `rilixai` CLI.

| Variable | Required for | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Local `rilixai dry-run` (calls the model directly) | [OpenAI dashboard](https://platform.openai.com/api-keys) |
| `RILIXAI_API_KEY` | Hosted runs (`rilixai push` / `trigger`) | rilixai dashboard |
| `RILIXAI_API_BASE_URL` | Hosted runs | rilixai dashboard — looks like `https://<id>.execute-api.us-east-2.amazonaws.com/prod/` |

For hosted runs you don't need `OPENAI_API_KEY` on your machine — it's
bound at the rilixai *project* level, and each sandbox container that
spawns inherits it as an env var. Set it once in the rilixai dashboard
and forget about it locally.

Per-recipe knobs like model choice, dataset size, and retrieval mode are
the fields of each recipe's `*SandboxConfig` in `rilixai_spec.py`, passed
as a JSON `--config` to `dry-run` / `trigger`.

## What runs where

Every recipe is driven by the top-level `rilixai` CLI:

| Command | Where it runs | When to use |
|---|---|---|
| `rilixai dry-run` | Your laptop | One case with the seed candidate — confirm wiring before pushing |
| `rilixai push` + `trigger` + `status` + `pull` | rilixai hosted Modal sandbox | Real optimization runs at scale, scheduled retraining, sharing optimized prompts across teams |

`dry-run` uses your `OPENAI_API_KEY` directly and prints to your
terminal. The hosted path packages the recipe into a Modal image, queues
the run through rilixai's API, and writes the optimized prompts back to
your rilixai project so any service of yours can fetch them at runtime.

## Development

```bash
uv sync --all-packages --group dev
uv run python -m pytest -q     # all recipes' tests
uv run ruff check
uv run ruff format --check
uv run python -m mypy
```

## rilixai dependency

Pinned via `git+ssh` at a specific main-branch SHA until a fresh
rilixai is published to PyPI. See each recipe's `pyproject.toml`
for the current pin.
