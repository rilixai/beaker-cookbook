# beaker-cookbook
<div align="center">
  <img width="227" height="225" alt="Beaker" src="https://github.com/user-attachments/assets/174d316a-24c0-4832-9f93-9c0a77a19433" />

  **[withbeaker.ai](https://withbeaker.ai)** · [app.withbeaker.ai](https://app.withbeaker.ai) · [beaker-sdk on PyPI](https://pypi.org/project/beaker-sdk/)
</div>

*Standalone agent recipes — each one a runnable agent plus a local
evaluation — covering different agent shapes, tasks, and production setups.
They are the reference systems we optimize with
[Beaker](https://withbeaker.ai), and a good starting point for wiring your
own agent into it.*

## Recipes

| Recipe | What it is | Keys |
|---|---|---|
| [`harvey_lab/`](harvey_lab/) | Legal-research agent over the Harvey LAB corpus, rubric-graded by an LLM judge | `OPENROUTER_API_KEY` (optional `GITHUB_TOKEN` to fetch the corpus) |
| [`hotpotqa/`](hotpotqa/) | Multi-hop QA — PydanticAI tool-using agent (`retrieve_k` + `summarize`) | `OPENAI_API_KEY` |
| [`apex_agents/`](apex_agents/) | APEX-Agents professional knowledge-work tasks — ReAct toolbelt agent, LLM rubric judge | `HF_TOKEN` (gated dataset), `OPENAI_API_KEY` (agent), `GOOGLE_API_KEY` (Gemini judge) |
| [`appworld/`](appworld/) | AppWorld benchmark — ReAct code agent on the OpenAI Agents SDK, TGC/SGC eval | `OPENAI_API_KEY` |
| [`automationbench/`](automationbench/) | Zapier AutomationBench — verifiers tool-calling agent + filesystem `skills/` hook | `OPENAI_API_KEY` (or Anthropic / Gemini, see README) |

Each recipe is a self-contained uv project (own `pyproject.toml` +
`uv.lock`; there is no root workspace) with:

- a README with the canonical reproduce commands and expected scores,
- a console script exposing `run` (execute the agent, dump outputs) and
  `evaluate` (also score and write a summary),
- a hermetic test suite so you can verify the harness before spending
  anything on LLM calls,
- documented data splits (see each README for how they are defined) so a
  recipe can be optimized on one split and reported on a held-out one.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+ (3.13+ for
`automationbench`). Pick a recipe, install it, export the keys from the
table above, and follow its README:

```bash
cd hotpotqa                      # or harvey_lab / apex_agents / appworld / automationbench
uv sync --group dev
export OPENAI_API_KEY=sk-...
uv run hotpotqa evaluate --split test --test-size 20 --output-dir hotpotqa_run
```

Keys are read from the environment (`export ...`). `automationbench/` also
loads a `.env` from its recipe directory (`cp .env.example .env`).

## Development

Each recipe carries its own lint / type / test config, and CI runs the same
steps per recipe:

```bash
cd <recipe>
uv sync --group dev --locked     # fails if uv.lock is stale — run `uv lock` after editing pyproject.toml
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q
```
