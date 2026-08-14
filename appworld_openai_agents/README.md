# AppWorld × OpenAI Agents SDK

A self-contained baseline harness for the
[AppWorld](https://appworld.dev/) benchmark: ~750 day-to-day digital tasks
(send money, queue songs, order groceries) executed against a simulated world
of 9 apps and ~460 APIs, scored by programmatic evaluators. This recipe runs
AppWorld's own **OpenAI Agents SDK** baseline agent — vendored from the
[upstream repo](https://github.com/StonyBrookNLP/appworld)
([paper](https://arxiv.org/abs/2407.18901), ACL 2024 Best Resource Paper) — with
a capability-aware model layer so the same harness sweeps both reasoning
(GPT-5 family) and non-reasoning (gpt-4.1/4o) OpenAI models by config only.

## The agent

The upstream **function-calling MCP agent**, unmodified in behavior
(`src/appworld_openai_agents/vendored/`, provenance in
[`ATTRIBUTION.md`](ATTRIBUTION.md)):

1. **API prediction pass** (`api_predictor`) — before the agent loop, a
   separate LLM call reads the task instruction plus one-line descriptions of
   all ~460 APIs and predicts which APIs the task needs (high recall, capped
   at 20). Only those tools are exposed to the agent via an MCP tool filter.
   This is AppWorld's own implementation choice (`retrieve_apis: true` in the
   upstream config) and part of a faithful baseline — without it, the tool
   list would blow past model tool limits.
2. **Agent loop** — an OpenAI Agents SDK `Agent` connected to the AppWorld
   MCP server (spawned locally by the harness), driven with the upstream
   instructions + worked-example demos, `tool_choice: auto`,
   `parallel_tool_calls: true`, and `max_steps: 50` turns per task. The loop
   ends when the agent calls `supervisor__complete_task` (or the step budget
   runs out); the world state is saved for the evaluator.

What is **not** vendored: the AppWorld environment itself (apps, tasks,
databases, MCP/API servers, evaluator). That *is* the benchmark and comes
from the pinned `appworld` pip dependency. The vendored agent targets
AppWorld APIs newer than the last PyPI release, so `appworld` is pinned to
the exact commit the agent was vendored from
(`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`).

Code map:

| Path | What it holds |
|---|---|
| `src/appworld_openai_agents/vendored/openai_agents/` | the upstream agent: run loop, MCP wiring, api_predictor, LM wrapper |
| `src/appworld_openai_agents/vendored/common/` | upstream scaffold helpers (logger, usage tracker, predictor base) |
| `src/appworld_openai_agents/prompts/` | upstream prompt/demo files, byte-identical |
| `src/appworld_openai_agents/models.py` | the capability-aware model layer (reasoning vs standard param assembly) |
| `src/appworld_openai_agents/config.py` | runner-config assembly (translation of the upstream jsonnet config) |
| `src/appworld_openai_agents/cli.py` | `run` / `evaluate` (`appworld-openai-agents` console command) |
| `configs/` | two ready-to-run model configs: `gpt-5.6.json` (reasoning), `gpt-4.1.json` (standard) |

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory. Requires Python 3.12+.

```bash
cd appworld_openai_agents
uv sync --group dev

# One-time AppWorld setup (~1 min): unpack the environment, then fetch the
# benchmark data (all splits, via AppWorld's CLI — not HuggingFace) into ./data.
uv run appworld install
uv run appworld download data
```

Provider key (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # or put it in .env (see .env.example)
```

## Run

```bash
# Smoke run: first 3 dev tasks on the reasoning-model config (~5-10 min,
# roughly $0.5-2 depending on effort/model):
uv run appworld-openai-agents run \
    --model-config configs/gpt-5.6.json \
    --split dev --max-tasks 3

# Score it (TGC/SGC on exactly those tasks):
uv run appworld-openai-agents evaluate \
    --model-config configs/gpt-5.6.json \
    --split dev --max-tasks 3
```

Full-split runs drop `--max-tasks` (`evaluate` then prints AppWorld's full
report and saves it under `experiments/outputs/<experiment>/evaluations/`):

```bash
uv run appworld-openai-agents run --model-config configs/gpt-5.6.json --split test_normal
uv run appworld-openai-agents evaluate --model-config configs/gpt-5.6.json --split test_normal
```

Everything is also flag-addressable without a config file — `--model`,
`--reasoning-effort` (ignored for non-reasoning models), `--temperature`
(ignored for reasoning models), `--split`, `--max-tasks`, `--max-steps`,
`--max-output-tokens`, `--output-dir`, `--experiment-name`:

```bash
uv run appworld-openai-agents run --model gpt-4.1 --temperature 0 --split dev --max-tasks 3
```

## Switching models: reasoning vs non-reasoning

Swapping model families is a config/flag change only. Each model resolves to
a capability profile (`models.py`) whose `family` decides which parameters
are attached — unsupported ones are omitted, never sent-and-400'd:

| | reasoning (gpt-5.6, gpt-5.6-sol/-terra/-luna, ...) | standard (gpt-4.1, gpt-4o, ...) |
|---|---|---|
| `reasoning.effort` | sent (`none`…`max`, default `medium`) | omitted |
| `temperature` / `top_p` / `seed` | omitted (the API 400s on them) | sent (defaults: 0.0 / — / 100) |

Both families run over the OpenAI **Responses API** (the Agents SDK's native
OpenAI route — `type: openai` never falls back to LitellmModel), so reasoning
settings are honored. Unknown snapshot ids route by prefix (`gpt-5*`/`o*` →
reasoning); `--family` forces it.

**Reasoning models are non-deterministic** (no temperature/seed): run the
eval N times and report mean TGC/SGC ± variance. `--max-output-tokens` is a
per-request cap and reasoning tokens count toward it — keep it high (the
reasoning config ships with 65 536) so answers don't get truncated.

## Data, splits, and metrics

`appworld download data` ships runnable data for **all** splits, including
pre-built initial state for the test tasks, and evaluation programs are
released for every split — so TGC/SGC are computable locally, no leaderboard
round-trip:

| Split | Tasks | Use |
|---|---|---|
| `dev` | 60 | default smoke/validation slice (full ground truth) |
| `test_normal` | 168 | headline baseline slice |
| `test_challenge` | 417 | headline baseline slice (harder, unseen apps) |

Only setup programs and reference solutions are withheld for test splits.
Don't do error analysis or tuning on test, and never surface ground truth to
the agent (the harness never does).

Metrics ([details](https://appworld.dev/)): **TGC** (Task Goal Completion —
fraction of tasks where all goal checks pass) and **SGC** (Scenario Goal
Completion — all tasks of a scenario pass), plus robustness checks that
penalize collateral state changes.

## Development

```bash
uv run pytest -q                                            # hermetic tests, no network/LLM
uv run ruff check && uv run ruff format --check && uv run python -m mypy
```

## Next steps

- **Agent-owned `skills/` folder** (not implemented): the injection point is
  where the agent's instructions are assembled — `run_agent_on_task` in
  `vendored/openai_agents/run.py` renders `prompts/function_calling_agent/instructions.txt`
  into the system prompt (`agent.instructions = system_prompt`). A skills
  loader would compose extra guidance into that render, next to the app
  descriptions.
- Sweep the GPT-5 family across `--reasoning-effort` levels and plot
  TGC/SGC vs cost.

## Attribution

Vendored agent code and prompts © Stony Brook NLP, Apache-2.0 — see
[`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and per-file provenance in
[`ATTRIBUTION.md`](ATTRIBUTION.md).
