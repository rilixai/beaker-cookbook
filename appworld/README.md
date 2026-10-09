# AppWorld

[AppWorld](https://github.com/StonyBrookNLP/appworld) ([paper](https://arxiv.org/abs/2407.18901), ACL 2024 Best Resource Paper) is a benchmark where an agent operates a simulated world of 9 apps and 457 APIs to complete everyday tasks (send money, order things, manage playlists) for a supervisor. Each task is scored by deterministic evaluation code that checks the final state of the world, not by string matching or a judge model.

This recipe is a ReAct-style code agent built on the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). It has one tool, `execute_python`, backed by AppWorld's Python environment, where the apps are callable as `apis.<app>.<api>(...)`. Nothing is pre-selected: the agent discovers APIs at runtime through the `api_docs` app, like the paper's ReAct baselines. The agent's system prompt lives in `prompts/`. Beaker improves the agent by editing `code_agent.py` and `prompts/`.

> **The Beaker integration is already present in this recipe.** `.beaker/` holds working configs and integrations; there is nothing to write before pointing Beaker at it.

## Quick start

You need Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and [Git LFS](https://git-lfs.com/) (`apt-get install git-lfs && git lfs install`). Git LFS must come first: `appworld install` fails without the LFS objects.

```bash
cd appworld
export UV_GIT_LFS=1                 # without this the appworld databases arrive as pointer files
uv sync --group dev
uv run appworld install             # AppWorld's one-time setup
uv run appworld download data       # benchmark data, all splits (a few hundred MB)
export OPENAI_API_KEY=sk-...        # or copy .env.example to .env

# Smoke run: 3 dev tasks
uv run appworld-openai-agents-sdk run --config configs/model.toml --split dev --max-tasks 3

# Score it (prints TGC and SGC)
uv run appworld-openai-agents-sdk evaluate --config configs/model.toml --split dev --max-tasks 3
```

The smoke run takes a minute or two and costs well under $1. Full splits take hours and tens of dollars.

Per task: AppWorld opens a fresh world, the agent submits Python chunks and reads the output, and the task ends when the agent calls `apis.supervisor.complete_task(...)` or runs out of steps (`--max-steps`, default 50). Predictions are written to `experiments/outputs/<experiment-name>/`.

## Tasks, scenarios, and splits

- **Task**: one instruction with its own pass/fail evaluation, e.g. "add the songs from my last playlist to a new playlist called X". Task IDs look like `82e2fac_1`.
- **Scenario**: a group of 3 variants of the same underlying problem (`82e2fac_1`, `82e2fac_2`, `82e2fac_3`). The ID prefix is the scenario ID.

| Split | Tasks | Scenarios | Used for |
|---|---|---|---|
| `train` | 90 | 30 | optimization |
| `dev` | 57 | 19 | evaluation |
| `test_normal` | 168 | 56 | not used by Beaker; baseline runs only |
| `test_challenge` | 417 | 139 | not used by Beaker; baseline runs only |

All splits run and score locally. Don't tune or do error analysis on the test splits.

## Metrics

- **TGC** (Task Goal Completion): the fraction of tasks that pass. A task passes only if every one of its requirements passes.
- **SGC** (Scenario Goal Completion): the fraction of scenarios where all 3 variants pass.

## Beaker integrations

Two agents, differing in what one Beaker case is and how it is scored. Both run from `appworld/` with `uv run beaker`.

| | Scenario agent | Task agent |
|---|---|---|
| Integration | `app_world` (`.beaker/beaker.yaml`) | `appworld_tgc` (`.beaker/tgc.yaml`) |
| Case | 1 scenario (3 tasks) | 1 task |
| Objective | `0.8 * TGC + 0.2 * SGC` | 1 if the task passes, else 0 |
| Full dataset | `appworld-sgc-full`: 30 train + 19 dev cases | `appworld-tgc-full`: 90 train + 57 dev cases |
| Upload | `uv run python .beaker/upload_dataset.py --full` | `uv run python .beaker/upload_dataset.py --tgc` |
| CLI | `uv run beaker` | `uv run beaker --config-file .beaker/tgc.yaml` |

Both draw on the same 147 train and dev tasks. The scenario objective gives a graded signal: TGC rewards partial progress, while SGC alone would be zero whenever any variant fails.

`upload_dataset.py` without a flag uploads a quick-start set (`appworld-sgc-quickstart`): the first 4 train scenarios, 3 for optimization and 1 held out. Hosted datasets change only when uploaded, so pass the dataset revision explicitly to smoke and launch. Uploading `--full` replaces the earlier revision under the same name; existing runs keep their original immutable revisions.

Each evaluator requirement becomes a pass/fail check grouped by task ID, so the optimizer sees which requirements failed.

**Editable scope:** `code_agent.py` and `prompts/`. Scoring, bootstrap code, model config, and vendored code are fixed. Integration internals are in [`.beaker/README.md`](.beaker/README.md).


## Beaker optimization results

Beaker independently optimized eight models on the full scenario dataset. Each
row compares the starting agent with that model's selected optimized agent on
held-out scenarios.

| Model | Starting score | Optimized score | Gain (percentage points) | Optimized cost / scenario |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 95.1% | **100.0%** | +4.9 | $0.448 |
| DeepSeek V4.1 Flash | 86.3% | 96.3% | +10.0 | $0.118 |
| MiMo-V2.6-Flash | 78.9% | 96.1% | +17.2 | $0.027 |
| Claude Haiku 5.5 | 66.0% | 96.1% | +30.2 | $0.033 |
| GPT-6 Luna | 92.3% | 93.5% | +1.2 | $0.011 |
| Kimi K3 | 83.5% | 93.2% | +9.6 | $0.414 |
| GLM 5.3 Flash | 36.0% | 89.3% | +53.3 | $0.033 |
| Gemini 3.5 Flash Lite | 44.9% | 87.7% | +42.8 | $0.070 |

The original GPT-6 Astra setup with high reasoning scored **100% at $0.624 per
scenario**. Beaker improved Gemini 3.8 Flash from **95.1% to 100%**, matching
Astra's score at **28.2% lower inference cost**. MiMo improved from **78.9% to
96.1%** at **$0.027 per scenario**, delivering a score within 3.9 percentage
points of Astra at **95.7% lower inference cost**.

The improvements came from concrete changes to agent behavior, including:

- Separating collection filters from item filters when selecting songs or other
  items within collections.
- Returning computed numeric answers as native numbers without extra formatting.
- Omitting unsolicited completion answers for requests that only require an action.

**How these scores were measured:** optimization used 30 training scenarios
(90 tasks); evaluation used 19 held-out scenarios (57 tasks) from AppWorld's
`dev` split. Scores are the combined objective, `0.8 × TGC + 0.2 × SGC`, described
[above](#metrics). Gemini 3.8 Flash was evaluated once per scenario; the other
models were evaluated twice, with scores averaged across repetitions. These are
recorded demo results; small differences can reflect model variability. Cost is
average model inference spend per scenario evaluation, including all three task
variants, and excludes the cost of optimization. Kimi K3 and Gemini 3.5 Flash
Lite used earlier integration builds and optimization playbooks.

**Try it yourself:** follow the [quick start](#quick-start), then use the scenario
integration and upload the full dataset with
`uv run python .beaker/upload_dataset.py --full`, as described in
[Beaker integrations](#beaker-integrations). The default quick-start dataset
checks the optimization workflow on a small set; these results use the full
dataset.

## Models

`configs/model.toml` has one block per model; `--model <name>` picks one. The default is `gpt-6-luna` with low reasoning. Or use flags:

```bash
uv run appworld-openai-agents-sdk run --model gpt-6-luna --reasoning-effort low --split dev --max-tasks 3
uv run appworld-openai-agents-sdk run --model gpt-4.1 --temperature 0 --split dev --max-tasks 3
```

Reasoning models (GPT-5/6, o-series) take `--reasoning-effort` and never receive `temperature`/`top_p`/`seed`. Standard models (GPT-4.1, GPT-4o) take `--temperature`/`--top-p` and never receive `reasoning`. Passing the wrong flag for a model is an error. `--max-output-tokens` defaults to 65,536 for reasoning models (reasoning tokens count toward it) and 16,384 otherwise.

In hosted Beaker runs, models route through Beaker's inference gateway, which controls reasoning and sampling settings.

Reasoning models are non-deterministic, so for a citable number run the evaluation several times and report mean ± variance.

## Code map

| Path | What it holds |
|---|---|
| `src/appworld_openai_agents_sdk/cli.py` | `run` / `evaluate` subcommands and flags |
| `src/appworld_openai_agents_sdk/models.py` | model layer (reasoning vs standard) |
| `src/appworld_openai_agents_sdk/code_agent.py` | the agent: `execute_python` tool and Agents SDK loop |
| `src/appworld_openai_agents_sdk/runner.py` | entry point |
| `src/appworld_openai_agents_sdk/prompts/` | agent instructions, adapted from upstream's ReAct prompt |
| `src/appworld_openai_agents_sdk/vendored/` | upstream logging helpers (Apache-2.0) |
| `configs/` | example model config |
| `.beaker/` | Beaker configs, integrations, and dataset upload |

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run pytest -q
```

## Attribution

Parts of this recipe are vendored from AppWorld (Apache-2.0). See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).
