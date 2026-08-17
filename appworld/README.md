# AppWorld

[AppWorld](https://github.com/StonyBrookNLP/appworld) is a benchmark where an
agent operates a simulated world of 9 apps and 457 APIs to complete everyday
tasks (send money, order things, manage playlists) for a supervisor. Every
task is scored by real evaluation code, not string matching. It received the ACL
2024 Best Resource Paper award ([arXiv:2407.18901](https://arxiv.org/abs/2407.18901)).

This recipe is a ReAct-style code agent built on the OpenAI Agents SDK: the
agent gets one tool, `execute_python`, backed by AppWorld's own Python
environment, where the apps are callable as `apis.<app>.<api>(...)`. Nothing
is pre-selected for it — it discovers the APIs it needs at runtime through the
`api_docs` app, like the AppWorld paper's ReAct baselines. A small model layer
on top lets you point it at any OpenAI model — reasoning (GPT-5 family) or not
(GPT-4.1/4o) — by changing a flag or a TOML file.

## Quick start (< 5 minutes)

You need Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/) (`apt-get install git-lfs && git lfs install`).
Git LFS must come first: the pinned `appworld` dependency ships its databases
as LFS objects, and `appworld install` fails without them. From this folder:

```bash
export UV_GIT_LFS=1                 # uv skips LFS smudge by default; without this the
                                    # appworld databases arrive as pointer files
uv sync --group dev                 # install (appworld + openai-agents, pinned)
uv run appworld install             # AppWorld's one-time self-setup
uv run appworld download data       # benchmark data, all splits (~few hundred MB)
export OPENAI_API_KEY=sk-...        # or: cp .env.example .env and fill it in

# Smoke run: 3 dev tasks (add --model gpt-4.1 etc. to pick another block of configs/model.toml)
uv run appworld-openai-agents run --config configs/model.toml --split dev --max-tasks 3

# Score it (prints TGC/SGC):
uv run appworld-openai-agents evaluate --config configs/model.toml --split dev --max-tasks 3
```

The 3-task smoke run takes a minute or two and costs well under $1, depending
on the model and reasoning effort. The real baselines are `--split test_normal` (168 tasks) and
`--split test_challenge` (417 tasks) — expect hours and tens of dollars per
full split.

## What actually runs

Per task (`src/appworld_openai_agents/code_agent.py`):

1. **A fresh AppWorld world is opened** for the task, with an in-process
   Python interpreter where the apps are callable as `apis.<app>.<api>(...)`.
   Variables persist across steps.
2. **The agent loop runs on the Agents SDK**: an `Agent` with instructions
   adapted from upstream's ReAct prompt (task, supervisor identity, app
   descriptions, one worked demo) and a single `execute_python` tool. Each
   turn the model submits a code chunk; the printed output (or traceback)
   comes back as the observation. No API list is pre-filled — the agent looks
   things up with `apis.api_docs.show_api_descriptions(...)` etc. as it goes.
3. **A task ends** when the agent runs `apis.supervisor.complete_task(...)`
   in code, or when the 50-step budget (`--max-steps`) runs out.
4. **Predictions are written** under `./experiments/outputs/<experiment-name>/`
   in the format `appworld evaluate` expects. `evaluate` prints **TGC**
   (Task Goal Completion) and **SGC** (Scenario Goal Completion).

## Data and splits

`appworld download data` fetches everything locally. All four splits run and score locally; the evaluation
programs are public even for the test sets (only setup programs and reference
solutions are withheld):

| Split | Tasks | Use |
|---|---|---|
| `dev` | 57 | default smoke/validation slice (full ground truth) |
| `test_normal` | 168 | main baseline |
| `test_challenge` | 417 | main baseline (harder) |
| `train` | 90 | training/demos |

Don't tune or do error analysis on the test splits, and never show the agent
withheld ground truth.

## Switching models

Whether a model is a reasoning model is detected from its name
(`src/appworld_openai_agents/models.py`: GPT-5 / o-series → reasoning,
everything else → standard), and that decides which parameters get sent.
Parameters a model doesn't support are simply not sent — no
send-and-catch-400:

- **reasoning** (GPT-5 family: `gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`,
  `gpt-5.6-luna`, ...): sends
  `reasoning = {"effort": none|minimal|low|medium|high|xhigh|max}`; never
  sends `temperature`/`top_p`/`seed` (the API rejects them with a 400).
- **standard** (`gpt-4.1`, `gpt-4o`, ...): sends `temperature`/`top_p` as
  usual; never sends a `reasoning` field.

If the auto-detection ever guesses wrong for a new model name, set
`family = "reasoning"` or `family = "standard"` in the TOML config to
override it.

Both families use the OpenAI **Responses API** — the Agents SDK default for
native OpenAI models, and the only place reasoning settings are honored.

[`configs/model.toml`](configs/model.toml) has one block per model, each
holding just the parameters that model supports; `--model <name>` picks the
block (the file's `default` is `gpt-5.6`). Or skip the config and use flags:

```bash
uv run appworld-openai-agents run --model gpt-5.6 --reasoning-effort high --split dev --max-tasks 3
uv run appworld-openai-agents run --model gpt-4.1 --temperature 0 --split dev --max-tasks 3
```

(`--reasoning-effort` only works with reasoning models and `--temperature`
only with standard ones — passing one to the wrong kind of model is an error.)

`--max-output-tokens` caps output tokens per request. **Reasoning tokens
count toward that cap**, so reasoning models default to a much larger one
(65,536 vs 16,384) to avoid truncated trajectories. There is no per-task
cost cap — watch the smoke run before scaling up.

> **Determinism note:** reasoning models accept no `temperature`/`seed`, so
> runs are non-deterministic. For a citable number, run the eval N times and
> report mean TGC/SGC ± variance.

## Code map

| Path | What it holds |
|---|---|
| `src/appworld_openai_agents/cli.py` | `run` / `evaluate` subcommands and all flags |
| `src/appworld_openai_agents/models.py` | the model layer (`ModelProfile`, reasoning vs standard) |
| `src/appworld_openai_agents/code_agent.py` | the agent: `execute_python` tool + Agents SDK loop |
| `src/appworld_openai_agents/runner.py` | entry point (max_steps=50, random_seed=100) |
| `src/appworld_openai_agents/prompts/react_code_agent/` | agent instructions, adapted from upstream's ReAct prompt (Apache-2.0) |
| `src/appworld_openai_agents/vendored/` | upstream logging helpers, verbatim modulo imports (Apache-2.0) |
| `configs/` | the example model config |

## Next steps (not implemented yet)

- **Agent-owned `skills/` folder.** The injection point is where the agent's
  system prompt is assembled: `render_instructions` in
  `src/appworld_openai_agents/code_agent.py` renders
  `prompts/react_code_agent/instructions.txt` into `agent.instructions`. A
  future `skills/` directory would be loaded there and appended to the
  rendered instructions. Deliberately left out of this baseline.
