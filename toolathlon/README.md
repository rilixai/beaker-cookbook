# Toolathlon baseline

Run [Toolathlon](https://github.com/hkust-nlp/Toolathlon)'s **original agent**, migrated
in place to **OpenAI Agents 0.20.0**. The standalone CLI supports all 108 official
tasks through the upstream container runner. Three tasks also run natively without
Docker or application accounts in the standalone CLI. [Beaker onboarding](.beaker/README.md)
uses isolated Docker environments for every task, including the quickstart, with
native OTLP tracing. The Docker path still needs an end-to-end VM validation.

## Native quickstart

Requires Linux or macOS, Python 3.12+, [uv](https://docs.astral.sh/uv/), Git and
Node.js 22+ with npm. Preparation installs Python **3.12.11**, downloads the pinned
upstream checkout and applies the versioned SDK migration. Allow several GB of disk.

```bash
cd toolathlon
uv sync --group dev --locked
uv run toolathlon-baseline prepare

# Connect to real MCP servers without model calls or credentials.
uv run toolathlon-baseline check --output runs/alita-tools

export TOOLATHLON_OPENAI_BASE_URL=https://openrouter.ai/api/v1
export TOOLATHLON_OPENAI_API_KEY=your-key
uv run toolathlon-baseline run \
  --model anthropic/claude-sonnet-4.5 \
  --task find-alita-paper --output runs/alita-001

uv run toolathlon-baseline results runs/alita-001
uv run toolathlon-baseline tasks
```

The endpoint must support OpenAI **Chat Completions** with tool calls. A paid
endpoint incurs model costs. Keep credentials in environment variables. The default
budget is 100 agent steps and 1,800 seconds, including setup and evaluation;
change it with `--max-steps` and `--timeout`.

| Native `--task` | Tools |
|---|---|
| `find-alita-paper` (default) | arXiv, scholarly search, filesystem |
| `excel-market-research` | Excel, terminal, filesystem |
| `interview-report` | Word, filesystem |

These are upstream's three
[native quickstart tasks](https://github.com/hkust-nlp/Toolathlon/blob/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb/scripts/quick_start/quick_start_run.sh).
`tasks` lists all 108 tasks and their MCP/local-tool requirements. The official
evaluator checks the resulting files and application state.

Each output must be a **new directory**. Native runs get independent source, task,
configuration and workspace copies; installed dependencies are reused. Cache/output
paths cannot contain spaces or shell metacharacters because upstream builds shell
commands from them. `prepare` and native `run` accept `--cache PATH` (default
`.cache/toolathlon`). Cached source must match the reviewed migration.

Results are in `result.json`, logs in `run.log`, the original trajectory in
`artifacts/traj_log.json`, the evaluator verdict in `artifacts/eval_res.json` when
available, and generated files in `artifacts/workspace/`. Exit codes are **0** for
pass, **1** for an evaluated failure, **2** for execution/setup/timeout or incomplete
evaluation, and **130** for interruption. Missing verdicts remain unscored.
Upstream cost estimates may be zero for models absent from its price table.

## Full benchmark: all 108 tasks

Requires a dedicated **Linux host**, Docker or Podman, the Toolathlon task image,
local application services and credentials for the selected remote services.
Python dependencies alone do not provision applications or accounts.

Prepare a separate checkout before configuring application credentials:

```bash
uv run toolathlon-baseline prepare --cache "$PWD/.cache/toolathlon-full"
export TOOLATHLON_ENV="$PWD/.cache/toolathlon-full/upstream"
```

Follow upstream's [full setup](https://toolathlon.xyz/docs/test) and
[account guide](https://github.com/hkust-nlp/Toolathlon/blob/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb/global_preparation/how2register_accounts.md)
in that checkout:

1. Run `bash global_preparation/install_env_minimal.sh false` and configure any
   required host/container settings.
2. Select Docker or Podman in `configs/global_configs.py`, expose its socket and
   run `bash global_preparation/pull_toolathlon_image.sh`.
3. Configure credentials for the required Google, GitHub, Notion, W&B, Hugging Face,
   Snowflake and other services, outside cookbook Git.
4. Run `bash global_preparation/misc_configuartion.sh` and
   `bash global_preparation/deploy_containers.sh false` as documented upstream.
5. Verify the actual applications with
   `bash global_preparation/check_installation_containerized.sh`.

Back in this recipe:

```bash
uv run toolathlon-baseline check-env --environment "$TOOLATHLON_ENV"

# Select a task; repeat --task for a subset.
uv run toolathlon-baseline run \
  --environment "$TOOLATHLON_ENV" --task ab-testing \
  --model anthropic/claude-sonnet-4.5 --output runs/subset-001

# Run every task once the required services/accounts are ready.
uv run toolathlon-baseline run \
  --environment "$TOOLATHLON_ENV" --all --workers 10 \
  --model anthropic/claude-sonnet-4.5 --output runs/full-001
```

Set the same endpoint/key variables as above; the endpoint must be reachable from
task containers. `check-env` verifies the revision, config files, runtime and image;
it does not authenticate every service. The recipe calls upstream `run_parallel.py`
and `scripts/run_single_containerized.sh`, preserving resource conflict locks and
phased evaluation/reference isolation. It copies migrated code and the lockfile
into containers, where upstream `uv run` installs the selected dependencies.

Defaults are one worker and a 5,400-second per-task timeout. `--workers`, `--timeout`
and `--max-steps` override them. `--runner containerized` selects this path explicitly;
`auto` selects it for an environment, multiple tasks or `--all`. Native mode accepts
one quickstart task at a time. The Docker/services path still needs validation
on a configured benchmark host.

Full-run artifacts are under `artifacts/finalpool/<task>/`; `result.json` records
all requested tasks. Missing/invalid verdicts remain errors, and aggregate
`pass_rate` is populated only when every task has a valid verdict. Execution/cleanup
errors exit 2, evaluated failures exit 1, and all tasks passing exits 0. Cleanup is
scoped to containers mounted under this run's output directory.

Reset application state before another formal run using upstream's documented
`deploy_containers.sh` workflow. Task containers do not undo remote API changes.
Upstream uses host networking and mounts the container runtime socket, giving
substantial host control. Use a dedicated benchmark VM/host and application accounts.
Standalone native execution uses the current user's privileges. Beaker's per-case
Docker execution is described in [.beaker/README.md](.beaker/README.md).

## Configuration and reproducibility

The recipe starts from `scripts/formal_run_v0.json`, using the original single-turn
`TaskAgent`, `unified` provider, tools and task prompts. `--model-params params.json`
uses upstream's `TOOLATHLON_MODEL_PARAMS_FILE` mechanism. `--system-prompt FILE` is a
native-only override for `docs/agent_system_prompt.md`; its upstream workspace
placeholder `!!<<<<||||workspace_dir||||>>>>!!` remains supported. Full runs use
prompts in the configured environment. Beaker applies its editable system prompt
inside every task container. Evaluators and references stay unchanged.
Keep OAuth state and run artifacts private; upstream logs can contain task config.

Source and task data are pinned to
[`9be8d8fe07a497b18ee61e3f2ae694e9797f39eb`](https://github.com/hkust-nlp/Toolathlon/tree/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb).
The versioned migration updates the original files and Python lock in place. One
upstream lock covers the agent group and the separate Excel server group; Excel's
old MCP requirement cannot share the agent's environment. The filesystem MCP and
its transitive packages reuse the pinned upstream npm lock. For full runs, `--image`
accepts an immutable image digest; summaries record the resolved image ID.

## Development

```bash
uv sync --group dev --locked
uv run ruff check
uv run ruff format --check
uv run python -m mypy
uv build --no-sources
```
