# AppWorld × OpenAI Agents SDK

Baseline harness for the **[AppWorld](https://github.com/StonyBrookNLP/appworld)**
benchmark (Trivedi et al., ACL 2024 Best Resource Paper,
[arXiv:2407.18901](https://arxiv.org/abs/2407.18901)): a simulated world of 9
apps / 457 APIs where an agent completes day-to-day tasks (send money, order
things, manage playlists) on behalf of a supervisor, scored by real evaluation
programs. This recipe vendors the benchmark's own **OpenAI Agents SDK** agent
and wraps it with a capability-aware model layer so you can sweep OpenAI
reasoning models (GPT-5 family) and non-reasoning models (GPT-4.1/4o) by
changing a flag or a TOML config only.

## Quick start (< 5 minutes)

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). From this folder:

```bash
uv sync --group dev                 # install (appworld + openai-agents, pinned)
uv run appworld install             # AppWorld's one-time self-setup
uv run appworld download data       # benchmark data (~all splits, few hundred MB)
export OPENAI_API_KEY=sk-...        # or: cp .env.example .env and fill it in

# Smoke run: 3 dev tasks with a reasoning model (or --config configs/gpt-4.1.toml)
uv run appworld-openai-agents run --config configs/gpt-5.6.toml --split dev --max-tasks 3

# Score it (prints TGC/SGC):
uv run appworld-openai-agents evaluate --config configs/gpt-5.6.toml --split dev --max-tasks 3
```

The 3-task smoke slice typically takes ~5–15 minutes of wall time (tasks are
long multi-step trajectories) and on the order of **$0.5–$2** depending on the
model and reasoning effort. Headline baselines are `--split test_normal` (168
tasks) and `--split test_challenge` (417 tasks); expect hours and tens of
dollars per full split.

## What runs, exactly

The vendored upstream scaffold (`src/appworld_openai_agents/vendored/`,
unchanged apart from import paths — see [ATTRIBUTION.md](ATTRIBUTION.md)):

1. **AppWorld servers** — the recipe starts the benchmark's API + MCP servers
   locally (`start_servers: true`); the agent's tools are the AppWorld APIs
   exposed over MCP.
2. **API predictor** — before the agent loop, a separate LLM pass reads the
   task instruction plus one-line descriptions of all 457 APIs and predicts
   the ≤20 APIs likely needed (`api_predictor.txt` prompt, 3 worked demos).
   Only those APIs are exposed to the agent as tools. This is upstream's
   `retrieve_apis: true` design and part of the faithful baseline.
3. **Agent loop** — an Agents SDK `Agent` with the upstream function-calling
   instructions + demos, `tool_choice: auto`, `parallel_tool_calls: true`, and
   a 50-turn budget (`--max-steps`). The task ends when the agent calls
   `supervisor__complete_task` or the budget runs out.
4. **Predictions** land under `./experiments/outputs/<experiment-name>/` in
   the exact format `appworld evaluate` expects; `evaluate` prints **TGC**
   (Task Goal Completion) and **SGC** (Scenario Goal Completion).

## Data and splits

`appworld download data` fetches everything locally (no HuggingFace, no
leaderboard round-trip). All four splits are locally runnable **and locally
scorable** — evaluation programs are released even for the test sets (only
setup programs and reference solutions are withheld):

| Split | Tasks | Use |
|---|---|---|
| `dev` | 60 | default smoke/validation slice (full ground truth) |
| `test_normal` | 168 | headline baseline |
| `test_challenge` | 417 | headline baseline (harder) |
| `train` | 90 | training/demos |

Don't do error analysis or tuning on the test splits, and never surface
withheld ground truth to the agent.

## Switching models (reasoning vs non-reasoning)

Every model entry declares a capability profile
(`src/appworld_openai_agents/models.py`), and a single `family` switch decides
which parameters are attached — unsupported parameters are omitted, never
sent-and-caught:

- **`family = "reasoning"`** (GPT-5 family: `gpt-5.6`, `gpt-5.6-sol`,
  `gpt-5.6-terra`, `gpt-5.6-luna`, ...): sends
  `reasoning = {"effort": none|minimal|low|medium|high|xhigh|max}`; never sends
  `temperature`/`top_p`/`seed` (the API rejects them with a 400).
- **`family = "standard"`** (`gpt-4.1`, `gpt-4o`, ...): sends
  `temperature`/`top_p` as usual; never sends a `reasoning` field.

Both families ride the OpenAI **Responses API** (the Agents SDK default for
native OpenAI models — the config's `type: openai` resolves to the SDK's
native OpenAI model, not LitellmModel, so reasoning settings are honored).

Two ready-to-run configs ship in [`configs/`](configs): `gpt-5.6.toml`
(reasoning, effort `medium`) and `gpt-4.1.toml` (standard, temperature 0).
Or skip configs entirely:

```bash
uv run appworld-openai-agents run --model gpt-5.6 --reasoning-effort high --split dev --max-tasks 3
uv run appworld-openai-agents run --model gpt-4.1 --temperature 0 --split dev --max-tasks 3
```

(`--reasoning-effort` is ignored for standard models, `--temperature` for
reasoning models. `--family` overrides the gpt-5*-based auto-detection.)

`--max-output-tokens` caps output tokens per model request; **reasoning tokens
count toward it**, so reasoning models default to a much larger cap (65,536 vs
16,384) to avoid truncated trajectories. There is no per-task cost cap in
upstream's openai_agents scaffold; watch the smoke slice before scaling up.

> **Determinism note:** reasoning models accept no `temperature`/`seed`, so
> runs are non-deterministic. For a citable number, run the eval N times and
> report mean TGC/SGC ± variance.

## Code map

| Path | What it holds |
|---|---|
| `src/appworld_openai_agents/cli.py` | `run` / `evaluate` subcommands and all flags |
| `src/appworld_openai_agents/models.py` | the capability-aware model layer (`ModelProfile`) |
| `src/appworld_openai_agents/runner.py` | the upstream jsonnet config, translated (max_steps=50, api_predictor, server setup) |
| `src/appworld_openai_agents/vendored/` | upstream agent scaffold, verbatim modulo imports (Apache-2.0) |
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
