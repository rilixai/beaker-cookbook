# AutomationBench + filesystem skills

Run an agent on business tasks in simulated apps such as Gmail, Sheets, Slack,
and Salesforce, then improve its prompts and skills with Beaker. This recipe uses
[AutomationBench-PI](https://github.com/xeophon/AutomationBench-PI/tree/858718b19e5d66c74451432783d34ee64a550a15),
a fork of Zapier's AutomationBench. You do not need accounts for the simulated
apps. Model calls use your provider account and incur charges.

The Beaker integration is already included in `.beaker/`.

## Quick start

Requires `uv`, Python 3.13 or later, and a model API key. The CLI loads `.env` automatically.

```bash
cd automationbench
uv sync --group dev --locked
export OPENAI_API_KEY=sk-...   # or copy .env.example to .env

# Run one training task with the same API tool interface as hosted Beaker
uv run automationbench-skills run --split train --limit 1 \
  --skills-dir skills --prompts-dir prompts --toolset api \
  --max-concurrent 1 --task-timeout 300 --output-dir runs/first-run

# Aggregate a finished (or partial) run directory
uv run automationbench-skills evaluate --output-dir runs/first-run
```

`run` writes one JSON per task (scores, trajectory, final world state) plus
`config.json` and `summary.json` to `--output-dir` (default
`runs/<split>-<timestamp>`). `evaluate` prints the pass rate (mean
`task_completed_correctly`) and mean `partial_credit`, per domain and overall.
`--task-timeout <seconds>` caps each rollout; a task that runs out of time is
still scored on the world it has changed so far. Setup and scoring can take
additional time. `evaluate` reads saved scores; it does not rerun tasks.
Use a new output directory for each experiment. A missing provider cost is not zero cost.

The CLI defaults to `--toolset zapier`; use `--toolset api` to match hosted runs.
Keep the model, tasks, toolset and limits fixed when comparing configurations.
Use train for development and reserve test for the final, frozen configuration.

## Understand the scores

The grader checks the simulated app state, not just the agent's final message.
`partial_credit` is the fraction of scored requirements met; `task_completed_correctly`
is 1 for full credit and 0 otherwise. The hosted objective combines **80% partial
credit and 20% full-task completion**.

Some checks are excluded, including conditions already satisfied at the start.
Breaking a protected condition can still fail. Inspect `assertion_results` and
`error` when investigating a score.

## Use Beaker

Fork this repository and connect it using the
[getting-started tutorial](https://www.withbeaker.ai/docs/getting-started/getting-started).
From `automationbench`, replace `YOUR_GITHUB_NAME` with your fork owner:

```bash
uv run beaker login
uv run beaker agent setup my-automationbench-agent \
  --integration-id automationbench-skills --repo YOUR_GITHUB_NAME/beaker-cookbook
uv run python .beaker/upload_splits.py
# Replace REVISION with the revision returned by the upload
uv run beaker run smoke --dataset automationbench-skills-quickstart@REVISION --strict
```

The uploader creates `automationbench-skills-quickstart` with **36 training and
18 test tasks**. Smoke checks setup and dataset loading without running a model.
Start an optimization run in Beaker with this integration and dataset revision.
**Checks** show task requirements and available grading observations; **OUTPUT**
summarizes changes to the simulated apps.

## Improve the agent

Edit `prompts/system.md` and the starter files under `skills/`, then rerun training
tasks. Each skill is a folder containing `SKILL.md` with YAML `name` and
`description` fields and a Markdown instruction body. The agent discovers and
reads skills through `list_skills` and `read_skill`.

Use `--no-skills` for a baseline with `prompts/system_no_skills.md` and no skill
tools. Keep the model, tasks, toolset and limits the same in both runs. Reserve
`--split test` for evaluating the configuration after development is complete.

## Choose a model

The default is `gpt-5.6-luna` with medium reasoning effort. Use `--model` and
`--reasoning-effort` to change it. Without a gateway or API override:

| Provider | Key | API |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | Responses |
| Claude (`claude-*`) | `ANTHROPIC_API_KEY` | Anthropic Messages |
| Gemini (`gemini-*`) | `GEMINI_API_KEY` | Gemini interactions |
| OpenRouter (`vendor/model`) | `OPENROUTER_API_KEY` | Chat Completions |

For an OpenAI-compatible gateway, add `--base-url`, `--api chat_completions`,
and `--api-key-var` to the run command. `--reasoning-enabled` supports OpenRouter
models with an on/off reasoning setting. See `uv run automationbench-skills run --help`
for all options.

Chat Completions defaults to 32,768 output tokens, up to three attempts per turn,
a 300-second request timeout, and a 600-second turn limit. `--task-timeout` can
stop a task earlier. Native provider limits differ.

## Dataset and compatibility

Standard splits contain **374 training and 104 test tasks** across six domains.
They exclude 122 task names based on an audit of the original benchmark, not an
audit of PI's repairs. See the [split guide](src/automationbench_skills/splits/README.md)
for selection details, exclusions, and held-out evaluation guidance.

Older name-only datasets must be regenerated with `.beaker/upload_splits.py`.
Custom rows must include frozen task data using the
[`CaseFixture` schema](.beaker/case_fixture.py); preserve original task contents
when rebuilding original-source cases. All fixtures run under the pinned PI runtime.

Rebuild the hosted integration after code changes. The spreadsheet grading fix
requires write history that older saved results do not contain. Rerun affected
cases; rescoring the old final state cannot recover that history.

## Development

```bash
uv sync --group dev --locked
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv build --no-sources
BEAKER_AGENT_KEY= uv run python -m pytest -q  # scripted or mocked clients
```

## Attribution

AutomationBench is MIT-licensed by Zapier, Inc. See [ATTRIBUTION.md](ATTRIBUTION.md)
and [LICENSE](LICENSE).
