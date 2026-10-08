# AutomationBench + filesystem skills

[AutomationBench](https://github.com/zapier/AutomationBench) ([paper](https://arxiv.org/abs/2604.18934)) is Zapier's benchmark for business automation agents: 600 public tasks in six domains (sales, marketing, operations, support, finance, hr). The agent works in a simulated workspace of SaaS tools (Gmail, Sheets, Slack, CRMs, ...) and is scored by deterministic assertions on the final state of that workspace.

This present recipe runs the benchmark one task at a time (`run_one`). The agent uses a system prompt in `prompts/`, and a `skills/` folder it reads through with two tools, `list_skills` and `read_skill`. Both are read from disk on every call, so Beaker can improve the agent by editing those files between rollouts. The optimizer itself is not part of this recipe.

> **The Beaker integration is already present in this recipe.** `.beaker/` holds a
> working `beaker.yaml` and `beaker_integration.py`; there is nothing to write
> before pointing Beaker at it.

## Quick start

```bash
cd automationbench
uv sync --group dev
export OPENAI_API_KEY=sk-...   # or copy .env.example to .env

# Smoke run: 3 test tasks with the seed skills and prompt
uv run automationbench-skills run --split test --limit 3 --skills-dir skills --prompts-dir prompts

# Aggregate a finished (or partial) run directory
uv run automationbench-skills evaluate --output-dir runs/<run-dir>
```

`run` writes one JSON per task (scores, trajectory, final world state) plus
`config.json` and `summary.json` to `--output-dir` (default
`runs/<split>-<timestamp>`). `evaluate` prints the pass rate (mean
`task_completed_correctly`) and mean `partial_credit`, per domain and overall.
`--task-timeout <seconds>` caps each rollout; a task that runs out of time is
still scored on the world it has changed so far.

## Baseline vs. skills

- **Vanilla**: `--no-skills` (or no `--skills-dir`). No skill tools; the
  system prompt is `prompts/system_no_skills.md`, which ships as the benchmark's
  own prompt.
- **Skills**: `--skills-dir skills`. The system prompt is `prompts/system.md`
  (the benchmark's prompt plus a paragraph on using the skills) and the agent
  has `list_skills()` / `read_skill(skill_id)` over the skills directory. An
  empty skills directory is fine.

A skill is a folder with a `SKILL.md` (YAML frontmatter `name`/`description`,
markdown body). Its ID is its path under `skills/`:

```text
skills/
  domains/{sales,marketing,operations,support,finance,hr}/SKILL.md
  apps/{gmail,google_sheets,google_drive,slack,salesforce,intercom,jira}/SKILL.md
```

The shipped skills are seeds (frontmatter plus a one-line stub); filling, adding,
splitting and merging them is the optimizer's job. `list_skills` returns every ID
with its description, no filtering. The seed apps are the most frequent ones in
the tasks' `zapier_tools`.

## Splits

We reviewed AutomationBench's task instructions, metadata, and grader code and
identified **122 cases with data-integrity defects**. The loader excludes these
cases before use, yielding **374 train and 104 test cases**, with case order and
split membership preserved. Exclusions address defective tasks or grading, not
case difficulty or agent performance.

See [exclusion reasons and source links](src/automationbench_skills/splits/EXCLUSIONS.md)
for the evidence and [split documentation](src/automationbench_skills/splits/README.md)
for counts, generation, and held-out evaluation.

`uv run python .beaker/upload_splits.py` uses the first six train and three test
cases per domain from the filtered splits: **54 cases (36 train / 18 test)**.
Hosted datasets change only when uploaded.

## Models

`--model` defaults to `gpt-5.6-luna` with `--reasoning-effort medium`. Routing
follows the benchmark (`vendored/model_setup.py`): `claude-*` goes to
Anthropic, `gemini-*` to the Gemini interactions API. Other names served
directly by OpenAI (no `--base-url` or `OPENAI_BASE_URL`) go to the Responses
API, because OpenAI rejects function tools on chat completions for gpt-5.4+
models unless `reasoning_effort` is `"none"`; a gateway stays on chat completions. `--api` overrides the choice.
`--reasoning-effort` maps to each API's reasoning setting. For a gateway:

```bash
uv run automationbench-skills run --split test --limit 3 \
  --model my-gateway/gemini-2.5-pro --base-url https://gateway.example/v1 --api-key-var GATEWAY_API_KEY
```

`gemini-*` names are routed to Gemini's native API even through a gateway
(upstream behavior). For a plain OpenAI-compatible gateway, use another model
name or pass `--api chat_completions`.

A `vendor/model` name (e.g. `z-ai/glm-5.3-flash`) routes to OpenRouter using
`OPENROUTER_API_KEY`; `--reasoning-enabled` toggles reasoning for models that
only expose an on/off switch.

## Inference limits and diagnostics

OpenAI-compatible Chat Completions and native OpenAI Responses calls default to
**16,384 output tokens**, **three total attempts per model turn**, and a
**300-second deadline** covering requests and retry backoff. SDK retries are
disabled so they cannot multiply that attempt budget. These defaults apply to
the CLI and the hosted Beaker client, including runs with a selected model.
Native Anthropic and Gemini clients retain their upstream settings.

Configure the CLI with `--max-output-tokens`, `--max-model-attempts`, and
`--model-request-timeout`, or the corresponding `ModelSpec` fields
`max_output_tokens`, `max_model_attempts`, and `model_request_timeout`. Chat calls
send `max_completion_tokens`; Responses calls send `max_output_tokens`. A lower
cap supplied in `extra_body` is honored, while a higher one cannot bypass the
configured cap. A cap covers hidden output and reasoning as well as visible text;
check truncation and task accuracy when changing it.

Every attempt prints a content-free `automationbench_model_attempt` JSON record
when it starts and when a response, error, retry, or deadline occurs. Records
include case/task and rollout identities, a model-turn ID, cap, attempt number,
elapsed time, and request/response IDs and usage when available. Prompts,
tool arguments, credentials, and response text are not printed by these logs.

Before any tools in a returned batch can run, the client verifies that every
`execute_tool.arguments` string contains one complete JSON object. Invalid JSON,
trailing garbage, and non-object values get at most one format-recovery retry,
within the same attempt/time budget. Persistent malformed arguments raise a tool
parse error so partial world state can still be scored. Valid Unicode is retained.
Rejected replies remain counted in usage/cost and in hosted per-request traces.

Two experiments are available without changing their existing defaults:

```bash
uv run automationbench-skills run --split test --limit 3 \
  --no-parallel-tool-calls --search-top-k 5
```

The equivalent `ModelSpec` options are `parallel_tool_calls=False` and
`search_top_k=5`. Serial calls consume more turns, and a smaller search result set
may hide a useful tool. Compare accuracy, turn counts, and latency with the same
settings for baseline and candidate evaluations. Rebuild the Integration image
from the updated cookbook commit before starting hosted evaluations.

## Reference numbers

Upstream reports strict pass rates (`task_completed_correctly`) of roughly
40–60% for frontier models on the public set. That is not the AutomationBench-AA
number, which adds guardrail and hidden-task components and uses a different
harness.

Historical result on the original, unfiltered 150-task test split (`--max-concurrent 16`, one seed;
skills arm uses `--skills-dir skills`, the seed skills):

| model | arm | pass_rate | partial_credit |
|---|---|---|---|
| `gpt-6-astra` (`--reasoning-effort max`) | no skills | 0.510 | 0.829 |
| `gpt-6-astra` (`--reasoning-effort max`) | skills | 0.507 | 0.837 |
| `z-ai/glm-5.3-flash` via OpenRouter (`--reasoning-effort max`) | no skills | 0.333 | 0.701 |
| `z-ai/glm-5.3-flash` via OpenRouter (`--reasoning-effort max`) | skills | 0.313 | 0.713 |
| `qwen/qwen3.8-flash` via OpenRouter (`--reasoning-effort default --reasoning-enabled`) | no skills | 0.413 | 0.795 |
| `qwen/qwen3.8-flash` via OpenRouter (`--reasoning-effort default --reasoning-enabled`) | skills | 0.447 | 0.802 |

### How a case is scored

There is no expected answer. Each task is a prompt against a simulated world
(Salesforce, Gmail, Sheets, ... as JSON records). After the agent acts, a list
of assertions is checked against that world
(`salesforce_campaign_member_exists`, `gmail_message_sent_to`, ...); each holds
or does not. Assertions already true before the agent acted don't count, so
doing nothing scores 0.

**Prefer `partial_credit` as the metric to optimize**, the share of assertions
that hold (0–1). It is the objective in `.beaker/beaker_integration.py` and
gives a much denser signal than `task_completed_correctly` (all assertions
hold), which is the strict pass rate. It is also possible to optimize a combination
of both.

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run pytest -q          # no network, scripted fake client
```

## Attribution

AutomationBench is MIT-licensed by Zapier, Inc. See [ATTRIBUTION.md](ATTRIBUTION.md)
and [LICENSE](LICENSE).
