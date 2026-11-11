# Toolathlon on Beaker

[Toolathlon](https://github.com/hkust-nlp/Toolathlon) (all 108 `tasks/finalpool` tasks) with its
upstream `toolathlon_default` agent loop (OpenAI Agents SDK), wired for Beaker harness optimization.

## Layout

| Path | What it is |
|---|---|
| `upstream/` | Toolathlon at commit `9be8d8fe07a497b18ee61e3f2ae694e9797f39eb`, verbatim except `assets/`, `vis_traj/` and the 24 files over 1 MB. It keeps its own pinned uv project (`upstream/pyproject.toml`, `upstream/uv.lock`). |
| `.beaker/upstream_assets.json` | The files over 1 MB (task archives, PDFs, datasets, `local_binary/github-mcp-server`). They are downloaded from the same upstream commit and checked against SHA-256 when a sandbox first needs them. They stay out of Git because Beaker downloads the whole cookbook repo, with a 100 MB compressed limit. |
| `.beaker/beaker_integration.py` | Rows, cases, `run_case`, `score_case`, and the editable targets. |
| `.beaker/toolathlon_setup.py` | Starts `dockerd`, pulls the task image, starts the self-hosted apps, and writes the credentials files. |
| `.beaker/upload_dataset.py` | Uploads one row per task, with a stable hash split: about 25% of tasks go to `test`. |

## How a case runs

Each case runs upstream's decoupled mode (`upstream/scripts/run_single_decoupled.sh`) inside the rollout sandbox. The
sandbox is a Modal VM, so Docker works there.

1. The task container (`lockon0927/toolathlon-task-image:1016beta`) runs the task's `preprocess`.
2. The same container serves every MCP tool the task needs over one SSE gateway.
3. The candidate's `toolathlon_default` loop runs in the sandbox. It talks to that gateway and to the rollout model
   through `inference_target(runtime)` (`TOOLATHLON_OPENAI_BASE_URL`, provider `unified`).
4. The container runs the task's `evaluation/main.py`. Its `eval_res.json` becomes the score: `1.0` only when Toolathlon
   reports `pass: true`.

Tasks that use self-hosted apps (poste.io mail, Canvas, WooCommerce, kind Kubernetes) start them with the upstream
`deployment/*/scripts/setup.sh`, once per sandbox.

## What Beaker optimizes

The editable targets are `upstream/utils/roles`, `upstream/utils/aux_tools`, `upstream/utils/system_prompts` and
`upstream/scripts/decoupled/host_agent_loop.py`, which together make up the agent loop, context management and prompts.
Candidates cannot edit task folders, per-task prompts, ground truth or evaluators.

## Secrets

66 tasks use real external accounts or the Serper search API. Set these values as encrypted secrets on the agent in
Beaker. They are written into upstream's untracked config files inside the sandbox and never committed:

`TOOLATHLON_SERPER_API_KEY`, `TOOLATHLON_GITHUB_TOKEN`, `TOOLATHLON_HUGGINGFACE_TOKEN`, `TOOLATHLON_WANDB_API_KEY`,
`TOOLATHLON_NOTION_INTEGRATION_KEY`, `TOOLATHLON_NOTION_INTEGRATION_KEY_EVAL`, `TOOLATHLON_NOTION_SOURCE_PAGE_URL`,
`TOOLATHLON_NOTION_EVAL_PAGE_URL`, `TOOLATHLON_SNOWFLAKE_ACCOUNT`, `TOOLATHLON_SNOWFLAKE_USER`,
`TOOLATHLON_SNOWFLAKE_ROLE`, `TOOLATHLON_SNOWFLAKE_PRIVATE_KEY` (PEM text), `TOOLATHLON_GCP_PROJECT_ID`,
`TOOLATHLON_GCP_SERVICE_ACCOUNT_JSON`, `TOOLATHLON_GOOGLE_OAUTH_CREDENTIALS_JSON`, `TOOLATHLON_GCP_OAUTH_KEYS_JSON`,
`TOOLATHLON_GOOGLE_CLOUD_CONSOLE_API_KEY`.

Use dedicated accounts. Upstream preprocess and evaluation delete and recreate repos, datasets, buckets and pages
(see `upstream/global_preparation/how2register_accounts.md`). A case whose task needs a missing secret fails at load
time and names the missing secret.

Two integrations share the same code:

- `toolathlon` (default): all 108 tasks, every secret above required.
- `toolathlon-no-secrets`: the 42 tasks that need no secret (`upload_dataset.py --no-secrets`).

## Limits

- Concurrency: cases in one sandbox share Docker and the app containers, and account tasks share the same accounts
  (see `upstream/tasks/finalpool/task_conflict.json`). `max_concurrency` defaults to 4.
- Time: upstream allows up to 5,400 s per task. Override with `TOOLATHLON_CASE_TIMEOUT_SECONDS` and
  `TOOLATHLON_MAX_STEPS`.
- Debugging: set `TOOLATHLON_DUMP_ROOT` to keep each case's dump (trajectory, gateway, preprocess and evaluation logs).
- License: the upstream repository has no license file. Confirm redistribution with the authors before publishing.

## Quickstart

```bash
cd toolathlon
uv run python .beaker/upload_dataset.py --no-secrets   # or without the flag, once secrets are set
```

## Checks

```bash
uv sync --group dev --locked
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q
```
