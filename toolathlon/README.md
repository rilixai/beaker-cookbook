# Toolathlon baseline

Run [Toolathlon](https://github.com/hkust-nlp/Toolathlon)'s original OpenAI Agents
SDK scaffold on **all 108 official tasks**, using Toolathlon's Linux
containerized runner. Three official quickstart tasks also run natively without
Docker or application accounts. Context/history management, tool-error recovery,
MCP tools, task instructions, initial workspaces, and evaluation come from Toolathlon.
The original agent is migrated in place to OpenAI Agents **0.20.0**. The optional
Beaker onboarding is documented in [.beaker/README.md](.beaker/README.md).

## Native quick start

Use Linux or macOS with Python 3.12+, [uv](https://docs.astral.sh/uv/), Git, and
Node.js 22+ with npm. `uv` installs **Python 3.12.11** for Toolathlon itself.
Preparation applies the reviewed SDK migration to the original upstream files
and upgrades their existing environment using the migrated `uv.lock`. The
standalone CLI and onboarding both execute that same original agent.
Preparation downloads the upstream repository and its full locked Python
environment; allow several GB of disk space. It needs no sudo, Docker, or
application accounts for these three tasks.

```bash
cd toolathlon
uv sync --group dev --locked
uv run toolathlon-baseline prepare

# Verify that the real MCP servers connect; no model calls or credentials needed.
uv run toolathlon-baseline check --output runs/alita-tools

# OpenRouter example; use the exact model ID exposed by your provider.
export TOOLATHLON_OPENAI_BASE_URL=https://openrouter.ai/api/v1
export TOOLATHLON_OPENAI_API_KEY=your-key
uv run toolathlon-baseline run \
  --model anthropic/claude-sonnet-4.5 \
  --task find-alita-paper \
  --output runs/alita-001

# Read the saved verdict without making model calls.
uv run toolathlon-baseline results runs/alita-001
```

The endpoint must accept OpenAI **Chat Completions** with tool calls. For a local
model server, set its base URL and a dummy API key if required. Credentials stay
in environment variables, never in the recipe's generated configuration.

Runs make paid model calls when using a paid endpoint. The default limit is
100 agent steps and 1,800 seconds for the whole task, including tool setup and
evaluation. Override with `--max-steps` and `--timeout`. Provider errors and arXiv
rate limits can consume this budget. Follow progress in `runs/alita-001/run.log`.

## Tasks and evaluation

`tasks` discovers the entire benchmark from the pinned `tasks/finalpool/`
directory, including each task's MCP/local-tool requirements:

```bash
uv run toolathlon-baseline tasks
# Or inspect a separately configured checkout:
uv run toolathlon-baseline tasks --environment /srv/toolathlon
```

The following three tasks have an additional account-free native launch mode:

| `--task` | Upstream task | Tools |
|---|---|---|
| `find-alita-paper` (default) | Find and download the Alita paper | arXiv, scholarly search, filesystem |
| `excel-market-research` | Edit the supplied market-research spreadsheets | Excel, terminal, filesystem |
| `interview-report` | Produce interview assessment documents | Word, filesystem |

These are the three tasks listed in the upstream
[native quickstart](https://github.com/hkust-nlp/Toolathlon/blob/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb/scripts/quick_start/quick_start_run.sh).
The evaluator checks actual files and task state. The recipe reports its verdict
without replacing it with a judge or inventing a reference answer.

Each `--output` must be a **new directory**. Every native run gets its own source tree,
task files, configuration, and workspace; the prepared dependency environment is
reused. Existing runs are never overwritten. Cache/output paths must not contain
spaces or shell metacharacters because upstream constructs shell commands from
these paths.

```text
runs/alita-001/
  result.json                 # Recipe summary and original evaluator verdict
  run.log                     # Agent, MCP, and evaluation stdout/stderr
  artifacts/
    traj_log.json             # Original Toolathlon trajectory and usage
    eval_res.json             # Original Toolathlon evaluator result, if produced
    workspace/                # Agent's working files
  source/                     # Per-run upstream code, task, and effective config
```

Exit codes: **0** = task passed; **1** = evaluated task failed; **2** = execution,
setup, timeout, or incomplete evaluation; **130** = interrupted. A missing or
`null` verdict stays unscored, rather than becoming a successful task. Inspect
`result.json`, `run.log`, and the trajectory when a run cannot complete.
Upstream cost estimates can be zero for model IDs absent from its price table;
use your provider's billing for actual spend.

## Full benchmark: all 108 tasks

The full runner accepts every task, any subset, or `--all`. It needs a dedicated
**Linux host** with Docker or Podman, the Toolathlon task image, deployed local
applications, and configured remote application accounts. Installing Python
dependencies alone cannot provision these accounts or application data.

Keep the configured environment separate from the immutable native cache. On
the Linux benchmark host, start a checkout at the same pinned revision:

```bash
export TOOLATHLON_ENV="$PWD/.cache/toolathlon-full/upstream"
git clone https://github.com/hkust-nlp/Toolathlon.git "$TOOLATHLON_ENV"
git -C "$TOOLATHLON_ENV" checkout 9be8d8fe07a497b18ee61e3f2ae694e9797f39eb
uv run toolathlon-baseline prepare --cache "$PWD/.cache/toolathlon-full"
```

Follow upstream's [full setup](https://toolathlon.xyz/docs/test) and
[account registration guide](https://github.com/hkust-nlp/Toolathlon/blob/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb/global_preparation/how2register_accounts.md)
inside that checkout. These are explicit administrator/setup operations, not
actions performed automatically by this recipe:

1. Run `bash global_preparation/install_env_minimal.sh false`. Upstream installs
   Python/Node dependencies and may install user-level development tools. Some
   container/Kubernetes host settings need administrator configuration.
2. Select Docker or Podman in `configs/global_configs.py`, make its socket
   available, and run `bash global_preparation/pull_toolathlon_image.sh`.
3. Configure the dedicated Google, GitHub, Notion, W&B, Hugging Face, Snowflake,
   and other service credentials required by the selected tasks, following the
   account guide. Values belong in the configured environment, not cookbook Git.
4. With credentials present, run `bash global_preparation/misc_configuartion.sh`
   and `bash global_preparation/deploy_containers.sh false` as upstream documents.
   Local services include Canvas, email, WooCommerce, and Kubernetes resources.
5. Run `bash global_preparation/check_installation_containerized.sh` to verify
   the actual MCP applications. Resolve application-specific failures first.

Back in the recipe, `check-env` checks the source revision, required config files,
container runtime, and installed image. It **does not authenticate every service**;
the upstream installation verification above supplies that evidence.

```bash
uv run toolathlon-baseline check-env --environment "$TOOLATHLON_ENV"

# One task outside the native quickstart:
uv run toolathlon-baseline run \
  --environment "$TOOLATHLON_ENV" \
  --task ab-testing \
  --model anthropic/claude-sonnet-4.5 \
  --output runs/ab-testing-001

# A subset: repeat --task.
uv run toolathlon-baseline run \
  --environment "$TOOLATHLON_ENV" \
  --task ab-testing --task find-alita-paper \
  --model anthropic/claude-sonnet-4.5 \
  --output runs/subset-001

# Every official task; account credentials and services must all be ready.
uv run toolathlon-baseline run \
  --environment "$TOOLATHLON_ENV" --all --workers 10 \
  --model anthropic/claude-sonnet-4.5 \
  --output runs/full-001
```

Set the same `TOOLATHLON_OPENAI_BASE_URL` and `TOOLATHLON_OPENAI_API_KEY` variables
as in the quickstart. The model endpoint must be reachable **from the task
containers**. The recipe calls upstream `run_parallel.py` and its original
`scripts/run_single_containerized.sh`; it retains the scheduler's resource
conflict locks and uses its `phased` execution mode to hide evaluator/reference
artifacts while the agent runs. It does not implement a different agent loop.

The default is one worker and a **5,400-second per-task timeout**. Override with
`--workers`, `--timeout`, and `--max-steps`. `--runner containerized` explicitly
chooses this path, including for a quickstart task. `auto` uses it whenever an
environment, multiple tasks, or `--all` is selected. Native mode accepts one of
the three quickstart tasks at a time.

Before another formal benchmark run, reset/restart the dedicated application
state using upstream `deploy_containers.sh` as its documentation specifies.
Task containers do not automatically undo remote Notion/GitHub/Google changes.

```text
runs/full-001/
  result.json                    # Counts and verdict for every requested task
  tasks.txt                      # Exact task selection
  eval_config.json               # Model/run configuration, without API keys
  run.log                        # Official scheduler log
  artifacts/finalpool/<task>/
    traj_log.json                # Original trajectory
    eval_res.json                # Original evaluator verdict, if produced
    status.json                  # Original preprocessing/execution/eval state
    ...                          # Workspace and task/container logs
```

Missing, `null`, and invalid verdicts remain errors in the complete task list.
The aggregate `pass_rate` is only populated when every requested task has a valid
verdict; incomplete runs cannot silently shrink the denominator. A full run with
any execution/cleanup error exits 2; otherwise any failed task exits 1, and all
tasks passing exits 0. The recipe performs additional cleanup scoped to the new
run's bind-mount paths, without removing application containers or other runs.

`--model-params` works with both runners. `--system-prompt` is a native override;
full runs use the task prompts in the explicitly configured environment. Keep
credentials, OAuth state, and full-run artifacts private: upstream artifacts can
include task-specific configuration.

## What the Linux sandbox does

The Linux host runs the scheduler and container runtime. Each task gets a fresh
container, initial workspace, and task-specific MCP configuration. Inside it,
Toolathlon runs preprocessing, starts the agent/tools, restores the protected
evaluation files, and runs the official evaluator. Results are bind-mounted to
the host; task containers are removed after execution. Local application
services and remote APIs have their own lifecycle outside this task container.

A container shares the host's Linux kernel; it is not a separate virtual
machine. Upstream uses host networking and mounts the Docker/Podman socket for
Kubernetes-related operations. That socket grants substantial host/container
control, so this is an evaluation environment for a dedicated Linux machine,
not a hardened isolation boundary for hostile code. Prefer a dedicated VM/host
and benchmark-only accounts; configured credentials and remote service state
also need their own access boundaries and resets.

## Configuration

The recipe starts from upstream `scripts/formal_run_v0.json`, uses the `unified`
provider and single-turn task mode, and keeps the original task system prompt.
The original agent uses its tool/history/context-management loop; this is not a
new agent implementation.

An optional prompt file replaces only `docs/agent_system_prompt.md` in the run
copy. The upstream placeholder `!!<<<<||||workspace_dir||||>>>>!!` expands to the
task workspace. Evaluators, task instructions, and references are unchanged:

```bash
uv run toolathlon-baseline run \
  --model anthropic/claude-sonnet-4.5 \
  --system-prompt prompts/my-system-prompt.md \
  --output runs/alita-custom-prompt
```

Use `--model-params params.json` to pass provider-specific request parameters
through Toolathlon's original `TOOLATHLON_MODEL_PARAMS_FILE` mechanism. For example,
`{"max_tokens": 8192}`. Check your endpoint's accepted parameters. The recipe
does not silently choose reasoning settings or translate model protocols.

`prepare` and native `run` accept `--cache PATH` (default `.cache/toolathlon`).
Prepare is repeatable: it verifies the source revision and installs from the same
Python and Node lockfiles. Don't edit the cached upstream source; use a prompt
override or change this recipe deliberately.

## Scope and reproducibility

Upstream is pinned to
[`9be8d8fe07a497b18ee61e3f2ae694e9797f39eb`](https://github.com/hkust-nlp/Toolathlon/tree/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb),
including its Python lockfile and official task data. The filesystem MCP package
matches upstream's `2025.7.1` version; this recipe also locks its Node dependencies.
Native MCP launch configurations use those installed executables, and the arXiv
wrapper receives the actual run source path instead of the container-only
`/workspace` path. Agent code and scoring scripts are copied without edits.

Native execution runs with your operating-system user's privileges. Its fresh
directories provide repeatable starting files, **not a security sandbox**. This
mode is for quickstart execution. Full benchmark execution uses the configured
Linux environment described above. The default container image is upstream's
tag; pass `--image` with a digest to fix it immutably. The full-run summary records
the resolved image ID. The container runner copies the migrated source and
lockfile into each container; upstream’s `uv run` installs those dependencies
there. The full Docker/services path still requires its own validation on a
configured benchmark host.

## Development

```bash
uv sync --group dev --locked
uv run ruff check
uv run ruff format --check
uv run python -m mypy
uv build --no-sources
```
