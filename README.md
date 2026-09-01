# beaker-cookbook
<div align="center">
  <img width="227" height="225" alt="Beaker" src="https://github.com/user-attachments/assets/174d316a-24c0-4832-9f93-9c0a77a19433" />

  <h3>Beaker, the autonomous AI engineer</h3>

  Beaker autonomously experiments with your AI agents — finding failures,
  testing fixes, and compounding the improvements that work.

  <a href="https://app.withbeaker.ai"><strong>Try for free →</strong></a> · <a href="https://withbeaker.ai">withbeaker.ai</a> · Docs (coming soon)
</div>

Every recipe in this cookbook is a real agent on a
real benchmark (including legal research, multi-hop QA, knowledge work, app control,
and enterprise automation) with a local eval you can run in minutes. Clone one, get a
baseline, then point Beaker at it and see what the first round of experiments
finds.

## Recipes

| Recipe | What it is | Keys |
|---|---|---|
| [`harvey_lab/`](harvey_lab/) | A junior-lawyer agent on Harvey's Legal Agent Benchmark: reads a case folder, writes the deliverables, gets graded criterion by criterion by an LLM judge. | `OPENROUTER_API_KEY` (optional `GITHUB_TOKEN` to fetch the corpus) |
| [`hotpotqa/`](hotpotqa/) | Multi-hop QA over Wikipedia. A PydanticAI agent with two tools, `retrieve_k` and `summarize`. | `OPENAI_API_KEY` |
| [`apex_agents/`](apex_agents/) | APEX-Agents: professional knowledge-work tasks. A ReAct agent with a toolbelt, graded against a rubric by an LLM judge. | `HF_TOKEN` (gated dataset), `OPENAI_API_KEY` (agent), `GOOGLE_API_KEY` (Gemini judge) |
| [`appworld/`](appworld/) | AppWorld: an agent that drives simulated apps by writing code. Built on the OpenAI Agents SDK, scored with TGC/SGC. | `OPENAI_API_KEY` |
| [`automationbench/`](automationbench/) | Zapier AutomationBench: a tool-calling agent on `verifiers` that can read a `skills/` folder — edit the skills, rerun, watch the score move. | `OPENAI_API_KEY` (or Anthropic / Gemini, see its README) |

Each recipe is a standalone uv project (own `pyproject.toml` + `uv.lock`, no
root workspace) and comes with:

- a README with the exact commands to reproduce it and the scores to expect
- a CLI with two commands: `run` (run the agent, save its outputs) and
  `evaluate` (run it and score it)
- offline tests, so you can check the whole harness before spending a cent on
  LLM calls
- documented data splits, so you can optimize on one and report on another

## Quick start

You need [`uv`](https://docs.astral.sh/uv/) and Python 3.12+ (3.13+ for
`automationbench`). Pick a recipe, install it, set your keys, and you're
running:

```bash
cd hotpotqa                      # or harvey_lab / apex_agents / appworld / automationbench
uv sync --group dev
export OPENAI_API_KEY=sk-...
uv run hotpotqa evaluate --split test --test-size 20 --output-dir hotpotqa_run
```

Keys are read from environment variables. `automationbench/` also reads a
`.env` in its own folder (`cp .env.example .env`). Every recipe README covers
the knobs: model, dataset size, retrieval mode, and so on.

## Development

Each recipe carries its own lint, type-check, and test config. CI runs the
same steps in every recipe folder:

```bash
cd <recipe>
uv sync --group dev --locked     # fails if uv.lock is out of date; run `uv lock` after changing pyproject.toml
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q
```

## Stay in the loop

More recipes are on the way. Star the repo to catch them, and follow
[@withbeaker on X](https://x.com/withbeaker) for Beaker news.
