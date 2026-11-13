# AutomationBench with skills

Test and improve an AI agent on business tasks in simulated apps such as Gmail,
Google Sheets, Slack, and Salesforce. A task might ask the agent to update a
customer record, check invoices, or send a follow-up email. The benchmark checks
what the agent changed to decide whether it completed the task.

This recipe uses [AutomationBench-PI](https://github.com/xeophon/AutomationBench-PI),
a version of Zapier's AutomationBench. It includes an agent, starter skills,
training and test tasks, and a ready-to-use Beaker integration. The apps and their
data are simulated; you do not need accounts for them. Model requests use your
provider account and incur its usual charges.

Start with one local task below, or follow [Use Beaker](#use-beaker) to optimize
the agent in Beaker.

## Run your first task

You need a checkout of this repository, `uv`, and an API key for your chosen
model. The project requires Python 3.13 or later. The default model is
`gpt-5.6-luna` with medium reasoning effort; see [Choose a model](#choose-a-model)
for other providers.

From the repository root:

```bash
cd automationbench
uv sync --locked
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY` to your key. The CLI loads this file
automatically. Then run one training task with the starter skills:

```bash
uv run automationbench-skills run \
  --split train --limit 1 \
  --skills-dir skills --prompts-dir prompts --toolset api \
  --max-concurrent 1 --task-timeout 300 \
  --output-dir runs/first-run
```

The agent reads the task, calls tools in the simulated apps, and receives a score.
Each task starts with its own app data. This command allows up to five minutes for
the task's model and tool loop; setup and scoring can take additional time.

These examples select `--toolset api`, the same tool interface used by hosted
Beaker runs. It lets the agent discover endpoints with `api_search` and call them
with `api_fetch`. The CLI also offers `--toolset zapier`, its default, which uses
tool search and execution. Keep the toolset fixed when comparing runs.

## Understand the results

The command prints a summary and saves these files in the output directory:

| File | Contents |
| --- | --- |
| `<task-name>.json` | Scores, model messages, tool calls, app state, and any error for one task. |
| `config.json` | The selected tasks and run settings. |
| `summary.json` | Average scores per domain and for the full run. |

Use a new output directory for each experiment. To summarize saved results again:

```bash
uv run automationbench-skills evaluate --output-dir runs/first-run
```

`evaluate` reads the saved scores. It does not call a model or run the tasks again.
For an unfinished run, it summarizes only the task files already written.

| Metric | Meaning |
| --- | --- |
| `partial_credit` | Fraction of scored requirements met, from 0 to 1. Useful for tracking progress on tasks with several steps. |
| `task_completed_correctly` | 1 when the task receives full credit; otherwise 0. |
| `pass_rate` | Average of `task_completed_correctly` across the completed tasks. |

A requirement is a check on the simulated apps, such as whether an email was sent
to the right person. A correct final chat message alone does not complete the task.
Some checks are excluded from the score, such as conditions already satisfied at
the start. Breaking a protected condition can count as a failure. Inspect each
task's `assertion_results` and `error` when investigating a low score.

A task that reaches its time limit keeps credit for work completed so far. Check
errors before comparing scores: a provider failure is different from an agent
choosing the wrong action. Cost is shown only when the provider reports it;
a missing cost is not zero cost.

## Improve the agent

The agent reads instructions from `prompts/system.md` and can read skill files
from `skills/`. A skill is a folder containing a `SKILL.md`. Its YAML frontmatter
sets the `name` and `description`; the Markdown body contains the instructions:

```text
skills/
  apps/gmail/SKILL.md
  apps/google_sheets/SKILL.md
  domains/sales/SKILL.md
```

The included skills are short starting points. Add procedures that help the agent
use an app or complete a type of task. The agent uses `list_skills` to see available
skills and `read_skill` to read one. Skills and prompts are read for each task, so
your edits take effect on the next run.

Compare the agent with and without skills on the same training tasks:

```bash
uv run automationbench-skills run \
  --split train --limit 3 --no-skills --prompts-dir prompts \
  --toolset api --max-concurrent 1 --task-timeout 300 \
  --output-dir runs/baseline

uv run automationbench-skills run \
  --split train --limit 3 --skills-dir skills --prompts-dir prompts \
  --toolset api --max-concurrent 1 --task-timeout 300 \
  --output-dir runs/with-skills
```

The baseline uses `prompts/system_no_skills.md` and has no skill tools. The skills
run uses `prompts/system.md` and enables the skill tools. Keep the model, tasks,
toolset, and limits the same when comparing these configurations.

Use training tasks to develop the agent. Once you finish editing and selecting a
configuration, run it with `--split test` for a held-out evaluation. Remove
`--limit` to run the whole split.

## Choose a model

Use `--model` to select a model your provider account can access. Set the matching
key in `.env`:

| Model selection | Key | API selected automatically |
| --- | --- | --- |
| Direct OpenAI model | `OPENAI_API_KEY` | Responses |
| `claude-*` | `ANTHROPIC_API_KEY` | Anthropic Messages |
| `gemini-*` | `GEMINI_API_KEY` | Gemini interactions |
| OpenRouter `vendor/model` name | `OPENROUTER_API_KEY` | OpenRouter Chat Completions |

The table applies when no gateway URL or API override is set. Use
`--reasoning-effort` for the model's supported reasoning level; `default` omits
that setting. `--reasoning-enabled` enables reasoning for OpenRouter models that
use an on/off setting.

For an OpenAI-compatible gateway, set its key in `.env` and pass its URL, model
name, and key variable explicitly. Replace the example values:

```bash
uv run automationbench-skills run \
  --split train --limit 1 --skills-dir skills --prompts-dir prompts \
  --toolset api --task-timeout 300 \
  --model YOUR_MODEL --base-url https://YOUR_GATEWAY/v1 \
  --api chat_completions --api-key-var GATEWAY_API_KEY
```

Chat Completions requests have a default output cap of 32,768 tokens and up to
three attempts per model turn. Each attempt has a 300-second timeout; the whole
turn, including retries, has a 600-second limit. `--task-timeout` can stop a task
earlier. Native provider APIs have different request limits.

For all command options, run `uv run automationbench-skills run --help`.

## Use Beaker

Beaker can run experiments that edit the skills, prompts, and agent code to
improve task scores. This recipe already includes the integration in `.beaker/`.
You do not need to write an integration.

Fork this repository, connect your fork to Beaker, and follow the
[Beaker getting-started tutorial](https://www.withbeaker.ai/docs/getting-started/getting-started)
for account and repository setup. Select the `automationbench-skills` integration.

From the `automationbench` directory in your fork, run `uv sync --locked` if you
have not installed the project yet. Then sign in and configure your agent.
Replace `YOUR_GITHUB_NAME` with the owner of your fork:

```bash
uv run beaker login
uv run beaker agent setup my-automationbench-agent \
  --integration-id automationbench-skills --repo YOUR_GITHUB_NAME/beaker-cookbook
uv run python .beaker/upload_splits.py
```

The uploader creates the `automationbench-skills-quickstart` dataset with
**36 training tasks and 18 test tasks**, spread across all six domains. It includes
the task data and grading requirements automatically. Use the dataset revision
returned by the upload when selecting a dataset for a run.

To check the integration and dataset before launching, replace `REVISION` below:

```bash
uv run beaker run smoke \
  --dataset automationbench-skills-quickstart@REVISION --strict
```

This checks setup and dataset loading. It does not run the agent or test model
quality. Start an optimization run in Beaker using the connected recipe and that
dataset revision. The hosted model defaults to GPT-5.6 Luna with medium reasoning.

In Beaker, **Checks** show which task requirements passed or failed. Failed checks
include available observations from grading to help explain the result.
**OUTPUT** summarizes changes to the simulated apps. Large outputs may be
shortened; scoring still uses the full returned app state and write history.
The optimization score combines 80% partial credit and 20% full-task completion.

## Dataset and evaluation scope

The recipe covers sales, marketing, operations, support, finance, and HR. Its
standard splits load **374 training tasks and 104 test tasks** from 600 public
tasks. The quickstart dataset is a smaller selection from these splits.

The selection excludes 122 task names based on an audit of the original benchmark.
That audit does not establish which issues remain in PI. See the
[dataset guide](src/automationbench_skills/splits/README.md) for selection details
and evaluation guidance. Compare runs on the same tasks, benchmark revision,
model, toolset, and limits.

## Further reading

- [Maintainer guide](MAINTAINING.md): integration internals, custom dataset format, and development checks.
- [Attribution and licenses](ATTRIBUTION.md): benchmark sources and copied provider code.
