# beaker-cookbook
<div align="center">
  <img width="227" height="225" alt="Beaker" src="https://github.com/user-attachments/assets/174d316a-24c0-4832-9f93-9c0a77a19433" />

  <h3>Beaker, the autonomous AI engineer</h3>

  Beaker autonomously experiments with your AI agents — finding failures,
  testing fixes, and compounding the improvements that work.

  <a href="https://app.withbeaker.ai"><strong>Try for free →</strong></a> · <a href="https://withbeaker.ai">withbeaker.ai</a> · Docs (coming soon)
</div>

This repo has the agents we use to test Beaker. Each one is a real, runnable
agent with its own local eval. Point Beaker at one agent, and see what the
first round of experiments finds.

## Recipes

| Recipe | What it is | Keys you need |
|---|---|---|
| [`harvey_lab/`](harvey_lab/) | Legal research agent on the Harvey LAB corpus. An LLM judge grades it against a rubric. | `OPENROUTER_API_KEY`. `GITHUB_TOKEN` is optional, for fetching the corpus. |
| [`hotpotqa/`](hotpotqa/) | Multi-hop QA. A PydanticAI agent with two tools: `retrieve_k` and `summarize`. | `OPENAI_API_KEY` |
| [`apex_agents/`](apex_agents/) | APEX-Agents knowledge-work tasks. A ReAct agent with a toolbelt, graded by an LLM judge. | `HF_TOKEN` (the dataset is gated), `OPENAI_API_KEY` (agent), `GOOGLE_API_KEY` (Gemini judge) |
| [`appworld/`](appworld/) | AppWorld benchmark. A ReAct code agent built on the OpenAI Agents SDK, scored with TGC/SGC. | `OPENAI_API_KEY` |
| [`automationbench/`](automationbench/) | Zapier AutomationBench. A tool-calling agent on `verifiers`, with a `skills/` folder it can read. | `OPENAI_API_KEY`, or Anthropic / Gemini keys (see its README) |

Every recipe is its own uv project with its own `pyproject.toml` and
`uv.lock`. There is no root workspace. Each one has:

- a README with the exact commands to reproduce it and the scores to expect
- a command-line tool with `run` (run the agent, save its outputs) and
  `evaluate` (run it and score it)
- tests that don't hit the network, so you can check the setup before you
  spend money on LLM calls
- data splits, described in the README, so you can tune on one and report on
  another

## Quick start

You need [`uv`](https://docs.astral.sh/uv/) and Python 3.12 or newer
(3.13 for `automationbench`). Go into a recipe, install it, set your keys,
and follow its README:

```bash
cd hotpotqa                      # or harvey_lab / apex_agents / appworld / automationbench
uv sync --group dev
export OPENAI_API_KEY=sk-...
uv run hotpotqa evaluate --split test --test-size 20 --output-dir hotpotqa_run
```

Keys are read from environment variables. `automationbench/` also reads a
`.env` file in its own folder (`cp .env.example .env`).

## Development

Each recipe has its own lint, type check, and test config. CI runs these
steps in every recipe folder:

```bash
cd <recipe>
uv sync --group dev --locked     # fails if uv.lock is out of date; run `uv lock` after changing pyproject.toml
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q
```

## Stay in the loop

Star this repo to hear about new recipes. Follow
[@withbeaker on X](https://x.com/withbeaker) for Beaker news.
