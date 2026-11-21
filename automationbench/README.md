# AutomationBench + filesystem skills

This recipe uses [AutomationBench-PI](https://github.com/xeophon/AutomationBench-PI), a fork of Zapier's [AutomationBench](https://github.com/zapier/AutomationBench) ([paper](https://arxiv.org/abs/2604.18934)), pinned to `858718b19e5d66c74451432783d34ee64a550a15`. Its loaders publish **800 tasks** across seven domains: sales, marketing, operations, support, finance, HR, and simple. The agent works in simulated SaaS services and PI's deterministic rubric scores the final world state.

This present recipe runs the benchmark one task at a time (`run_one`). The agent uses a system prompt in `prompts/`, and a `skills/` folder it reads through with two tools, `list_skills` and `read_skill`. Both are read from disk on every call, so Beaker can improve the agent by editing those files between rollouts. The optimizer itself is not part of this recipe.

> **The Beaker integration is already present in this recipe.** `.beaker/` holds a
> working `beaker.yaml` and `beaker_integration.py`; there is nothing to write
> before pointing Beaker at it.

Beaker displays assertion conditions as **Checks**, without Expected/Predicted
value pairs. Check names include the assertion type and complete parameter values,
resolving record IDs to readable labels when available. Descriptions include the
handler's first docstring line and the full condition parameters with original IDs;
scoring metadata (`scored`, `excluded`) is omitted from both names and descriptions.
Failed Checks show a readable summary of required conditions, recorded value
comparisons, and the latest evaluator observations captured during scoring.
Messages omit Python code and function traces, and explicitly note that the
diagnostic is partial; it does not infer a cause from the last comparisons alone.
**OUTPUT** shows app changes once per sample with size limits and omission markers.
Display output is reduced or omitted when the combined result approaches Beaker's
transport limit, preserving the complete scoring state before display details.
The full final state and spreadsheet write history remain in `context` for native
scoring. Older saved results lack this history; rerun affected spreadsheet cases
with the rebuilt integration rather than rescoring the old final state.

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

The loader retains **25 exclusions** from the previous recipe and restores the 97 excluded tasks whose definitions PI directly changed. This leaves **584 train and 191 test cases**. Existing split membership stays fixed; the 200 `simple` tasks add 150 train and 50 test cases. Direct edits do not guarantee that every old defect is resolved; this is a dataset selection policy, not a new full audit.

See [exclusion reasons and source links](src/automationbench_skills/splits/EXCLUSIONS.md)
and [split documentation](src/automationbench_skills/splits/README.md).

`uv run python .beaker/upload_splits.py` selects the first six train and three test
cases per domain: **63 cases (42 train / 21 test)**. It uploads under
`automationbench-pi-skills-quickstart` and records the pinned benchmark commit.
Rebuild the hosted integration and upload a fresh dataset before using this recipe;
old dataset assertions belong to the previous benchmark. This source change does
not update existing hosted datasets or agents.

Standard PI rows resolve by task name and must identify the pinned benchmark
commit. Frozen pilot rows also carry `runtime_fixture` with the task, scenario,
and matching hashes (see [the schema](.beaker/case_fixture.py)). Their saved task
data is preserved, including original-source examples. Both formats use PI API
tools and graders in hosted runs and the default CLI interface.

Standard rows and CLI runs share one scenario clock. An explicit world timestamp
takes precedence; otherwise, an authored current-date statement supplies the clock
(noon UTC for a date without a time). Tasks with neither use February 20, 2026 at
12:00 UTC. The host's current date is never used. Frozen pilot rows keep their saved
scenario fallback times.

## PI runtime

The recipe uses PI's native Verifiers V1 task and `NullHarness`. PI owns the
agent loop, MCP execution, turn limits, and retries. The Beaker adapter adds skill
tools, traces model/tool calls, and exports the final world and spreadsheet write
history for trusted scoring. Hosted runs and the CLI default to the API toolset.

## Models

`--model` defaults to `gpt-5.6-luna` with `--reasoning-effort medium`. The native
harness requires a Chat Completions-compatible endpoint. Use `--base-url` (or
`OPENAI_BASE_URL`) for a gateway. Separate Anthropic Messages, OpenAI Responses,
and Gemini Interactions adapters are no longer included.

```bash
uv run automationbench-skills run --split test --limit 3 \
  --model my-gateway/gemini-2.5-pro --base-url https://gateway.example/v1 --api-key-var GATEWAY_API_KEY
```

A `vendor/model` name routes to OpenRouter using `OPENROUTER_API_KEY` unless an
explicit base URL is supplied. `--reasoning-enabled` supports gateway models with
an on/off reasoning switch.

Hosted runs without a model choice can evaluate a model on another Chat
Completions-compatible endpoint, such as a self-hosted vLLM server. Set these on
the agent:

- `AUTOMATIONBENCH_MODEL`: the model name the endpoint serves.
- `AUTOMATIONBENCH_BASE_URL`: the endpoint's base URL, ending in `/v1`.
- `AUTOMATIONBENCH_API_KEY_VAR`: the name of the agent variable holding its key.
- `AUTOMATIONBENCH_EXTRA_BODY` (optional): a JSON object of extra request fields,
  for example `{"temperature": 0.8, "chat_template_kwargs": {"enable_thinking": false}}`.

A run's own model choice still wins. Endpoint calls are traced, but Beaker does
not price them.

## Execution limits

Each run allows 50 model turns by default (`--max-steps`). The default output cap
is 32,768 tokens, including reasoning. Python callers can set a lower cap or an
explicit reasoning budget through `ModelSpec.extra_body`. Provider-specific
sampling values must be supported by the selected endpoint.

`--task-timeout` bounds the agent's solve attempt. Completed writes are preserved
for scoring after a timeout. Provider failures request a case retry in hosted
Beaker runs. Request retries are managed by PI's native harness, not a separate
recipe retry loop. Rebuild the Integration image after updating the recipe.

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
or does not. PI excludes unchanged pre-satisfied assertions unless they explicitly opt into
scoring. A passing negative assertion earns credit only after the scored positive
requirements pass; failed negative guards still penalize the run. Inverse tasks
can explicitly reward an untouched world. Beaker uses this same PI score and
marks excluded checks as informational.

**Prefer `partial_credit` as the metric to optimize**, the share of assertions
that hold (0–1). It is the objective in `.beaker/beaker_integration.py` and
gives a much denser signal than `task_completed_correctly` (all assertions
hold), which is the strict pass rate. It is also possible to optimize a combination
of both.

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run python -m mypy
BEAKER_AGENT_KEY= uv run python -m pytest -q  # local HTTP/MCP servers, no paid model calls
uv build --no-sources
```

## Attribution

AutomationBench is MIT-licensed by Zapier, Inc. See [ATTRIBUTION.md](ATTRIBUTION.md)
and [LICENSE](LICENSE).
