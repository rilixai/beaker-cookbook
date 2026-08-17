# AppWorld

[AppWorld](https://github.com/StonyBrookNLP/appworld) is a benchmark where an
agent operates a simulated world of 9 apps and 457 APIs to complete everyday
tasks (send money, order things, manage playlists) for a supervisor. Every
task is scored by real evaluation code, not string matching. It won the ACL
2024 Best Resource Paper award ([arXiv:2407.18901](https://arxiv.org/abs/2407.18901)).

This recipe vendors the benchmark's own OpenAI Agents SDK agent so it runs
standalone, and adds a small model layer on top so you can point it at any
OpenAI model — reasoning (GPT-5 family) or not (GPT-4.1/4o) — by changing a
flag or a TOML file.

## Quick start (< 5 minutes)

You need Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/) (`apt-get install git-lfs && git lfs install`).
Git LFS must come first: the pinned `appworld` dependency ships its databases
as LFS objects, and `appworld install` fails without them. From this folder:

```bash
uv sync --group dev                 # install (appworld + openai-agents, pinned)
uv run appworld install             # AppWorld's one-time self-setup
uv run appworld download data       # benchmark data, all splits (~few hundred MB)
export OPENAI_API_KEY=sk-...        # or: cp .env.example .env and fill it in

# Smoke run: 3 dev tasks with a reasoning model (or --config configs/gpt-4.1.toml)
uv run appworld-openai-agents run --config configs/gpt-5.6.toml --split dev --max-tasks 3

# Score it (prints TGC/SGC):
uv run appworld-openai-agents evaluate --config configs/gpt-5.6.toml --split dev --max-tasks 3
```

The 3-task smoke run takes roughly 5–15 minutes (tasks are long, multi-step)
and costs on the order of $0.5–$2 depending on the model and reasoning
effort. The real baselines are `--split test_normal` (168 tasks) and
`--split test_challenge` (417 tasks) — expect hours and tens of dollars per
full split.

## What actually runs

The vendored upstream code (`src/appworld_openai_agents/vendored/`, unchanged
apart from import paths — see [ATTRIBUTION.md](ATTRIBUTION.md)) does four
things:

1. **Starts the AppWorld servers** locally. The agent's tools are the
   AppWorld APIs, exposed over MCP.
2. **Predicts which APIs the task needs.** Before the agent loop, a separate
   LLM call reads the task plus one-line descriptions of all 457 APIs and
   picks the ≤20 it will likely need. Only those become tools. This is
   upstream's `retrieve_apis: true` design, kept as-is for a faithful
   baseline.
3. **Runs the agent loop**: an Agents SDK `Agent` with upstream's
   instructions and demos, `tool_choice: auto`, `parallel_tool_calls: true`,
   and a 50-turn budget (`--max-steps`). A task ends when the agent calls
   `supervisor__complete_task` or runs out of turns.
4. **Writes predictions** under `./experiments/outputs/<experiment-name>/`
   in the format `appworld evaluate` expects. `evaluate` prints **TGC**
   (Task Goal Completion) and **SGC** (Scenario Goal Completion).

## Data and splits

`appworld download data` fetches everything locally — no HuggingFace, no
leaderboard round-trip. All four splits run and score locally; the evaluation
programs are public even for the test sets (only setup programs and reference
solutions are withheld):

| Split | Tasks | Use |
|---|---|---|
| `dev` | 60 | default smoke/validation slice (full ground truth) |
| `test_normal` | 168 | main baseline |
| `test_challenge` | 417 | main baseline (harder) |
| `train` | 90 | training/demos |

Don't tune or do error analysis on the test splits, and never show the agent
withheld ground truth.

## Switching models

Each model declares a `family` (`src/appworld_openai_agents/models.py`), and
that one switch decides which parameters get sent. Parameters a model doesn't
support are simply not sent — no send-and-catch-400:

- **`family = "reasoning"`** (GPT-5 family: `gpt-5.6`, `gpt-5.6-sol`,
  `gpt-5.6-terra`, `gpt-5.6-luna`, ...): sends
  `reasoning = {"effort": none|minimal|low|medium|high|xhigh|max}`; never
  sends `temperature`/`top_p`/`seed` (the API rejects them with a 400).
- **`family = "standard"`** (`gpt-4.1`, `gpt-4o`, ...): sends
  `temperature`/`top_p` as usual; never sends a `reasoning` field.

Both families use the OpenAI **Responses API** — the Agents SDK default for
native OpenAI models (the config's `type: openai` resolves to the SDK's own
OpenAI model class, not LitellmModel, so reasoning settings are honored).

Two ready-to-run configs ship in [`configs/`](configs): `gpt-5.6.toml`
(reasoning, effort `medium`) and `gpt-4.1.toml` (standard, temperature 0).
Or skip configs and use flags:

```bash
uv run appworld-openai-agents run --model gpt-5.6 --reasoning-effort high --split dev --max-tasks 3
uv run appworld-openai-agents run --model gpt-4.1 --temperature 0 --split dev --max-tasks 3
```

(`--reasoning-effort` is ignored for standard models, `--temperature` for
reasoning models. `--family` overrides the gpt-5*-based auto-detection.)

`--max-output-tokens` caps output tokens per request. **Reasoning tokens
count toward that cap**, so reasoning models default to a much larger one
(65,536 vs 16,384) to avoid truncated trajectories. One wrinkle: the API
predictor uses Chat Completions (upstream's choice), where reasoning models
take the cap as `max_completion_tokens`; the model layer handles that for
you. Upstream's scaffold has no per-task cost cap — watch the smoke run
before scaling up.

> **Determinism note:** reasoning models accept no `temperature`/`seed`, so
> runs are non-deterministic. For a citable number, run the eval N times and
> report mean TGC/SGC ± variance.

## Code map

| Path | What it holds |
|---|---|
| `src/appworld_openai_agents/cli.py` | `run` / `evaluate` subcommands and all flags |
| `src/appworld_openai_agents/models.py` | the model layer (`ModelProfile`, reasoning vs standard) |
| `src/appworld_openai_agents/runner.py` | upstream's jsonnet config, translated (max_steps=50, api_predictor, server setup) |
| `src/appworld_openai_agents/vendored/` | upstream agent code, verbatim modulo imports (Apache-2.0) |
| `src/appworld_openai_agents/prompts/` | upstream agent + api_predictor prompts and demos |
| `configs/` | the two example model configs (reasoning + standard) |

## Next steps (not implemented yet)

- **Agent-owned `skills/` folder.** The injection point is where the agent's
  system prompt is assembled: `run_agent_on_task` in
  `src/appworld_openai_agents/vendored/openai_agents/run.py` renders
  `prompts/function_calling_agent/instructions.txt` and sets
  `agent.instructions = system_prompt`. A future `skills/` directory would be
  loaded there and appended to the rendered instructions. Deliberately left
  out of this baseline.
