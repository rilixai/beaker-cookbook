# Toolathlon baseline

Run [Toolathlon](https://github.com/hkust-nlp/Toolathlon)'s original OpenAI Agents
SDK scaffold on its three official native quickstart tasks. The recipe calls
upstream `main.py`: context/history management, tool-error recovery, MCP tools,
task instructions, initial workspaces, and evaluation all come from Toolathlon.
There is no Beaker dependency or integration in this recipe.

## Quick start

Use Linux or macOS with Python 3.12+, [uv](https://docs.astral.sh/uv/), Git, and
Node.js 22+ with npm. `uv` installs **Python 3.12.11** for Toolathlon itself, in a
separate environment because its OpenAI Agents SDK is pinned to **0.0.15**.
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

| `--task` | Upstream task | Tools |
|---|---|---|
| `find-alita-paper` (default) | Find and download the Alita paper | arXiv, scholarly search, filesystem |
| `excel-market-research` | Edit the supplied market-research spreadsheets | Excel, terminal, filesystem |
| `interview-report` | Produce interview assessment documents | Word, filesystem |

These are the three tasks listed in the upstream
[native quickstart](https://github.com/hkust-nlp/Toolathlon/blob/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb/scripts/quick_start/quick_start_run.sh).
The evaluator checks actual files and task state. The recipe reports its verdict
without replacing it with a judge or inventing a reference answer.

Each `--output` must be a **new directory**. Every run gets its own source tree,
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

Both `prepare` and `run` accept `--cache PATH` (default `.cache/toolathlon`).
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
is a quickstart subset, not a full Toolathlon leaderboard evaluation. The full
benchmark also needs deployed services, application credentials, and task
isolation; follow the upstream [containerized setup](https://toolathlon.xyz/docs/test)
for that. A future Beaker integration can call this recipe separately.

## Development

```bash
uv sync --group dev --locked
uv run ruff check
uv run ruff format --check
uv run python -m mypy
uv run pytest -q
uv build --no-sources
```

Tests exercise the recipe's process lifecycle, run isolation, and failure
reporting without downloading Toolathlon or calling an LLM. They are not
benchmark results.
