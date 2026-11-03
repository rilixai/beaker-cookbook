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
The full final state remains in `context` for native scoring.

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
  domains/{sales,marketing,operations,support,finance,hr,simple}/SKILL.md
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

## Balanced 210-case demo

The larger demo selects **20 train and 10 test tasks from each of the seven domains**:
**140 train / 70 test**, or **210 total**. The 25 exclusions are applied first.
Selection stays within each frozen source split. Within each domain and split,
task-family quotas are proportional to the eligible pool, rounded by largest
remainder, with seed 0 used to select family members. Output retains source order.
Thus domains have equal weight, even though `simple` has more source tasks.
Small quotas cannot represent every task family; integer family allocations
remain within one sample of their proportional targets.

Generate the demo **locally, without uploading**:

```bash
uv run python -m automationbench_skills.data.demo --output-dir /tmp/automationbench-pi-demo210
```

This writes `train.jsonl`, `test.jsonl`, and `verification.json` with per-domain
and per-family counts, the benchmark revision, exclusion/overlap checks, and
file hashes. Existing demo files are not overwritten. The command does not need
Beaker credentials or invoke the Beaker CLI.

The uploader's optional `--profile demo` uses exactly the same selection and the
name `automationbench-pi-demo210`; its default remains the 63-case quickstart.
Run the uploader only when you intend to upload a dataset.

## PI runtime

PI removed the legacy Verifiers environment and model clients. This recipe runs
its unmodified MCP toolset in a separate in-process session for each rollout,
adds the skill tools, and calls the pinned PI rubric. Provider adapters retain
Chat Completions, Responses, Anthropic and Gemini routing. Model requests and MCP
tool calls emit Beaker spans directly. No legacy Verifiers instrumentation is used.

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

## Chat Completions limits

OpenAI-compatible Chat Completions calls use **32,768 output tokens** (a 24,576-token
reasoning allowance plus 8,192 for the answer), at most **three attempts per model
turn**, a **300-second timeout per attempt**, and a **600-second deadline** for the
whole turn, covering every attempt and the backoff between them. SDK retries are
disabled. These limits apply to the hosted Beaker gateway and local runs using
`--api chat_completions`. Direct OpenAI CLI
runs select native Responses by default; Responses, Anthropic, and Gemini clients
retain their upstream behavior.

The defaults are named constants in `src/automationbench_skills/clients.py`.
Reasoning counts toward the completion cap on most providers, so a reasoning
budget in `ModelSpec.extra_body` (OpenRouter's `{"reasoning": {"max_tokens": N}}`)
sets the cap to that budget plus the answer allowance; where the provider honors
the budget, a call stops reasoning and still answers. The CLI does not expose
`extra_body`, and a budget applies only to runs that call OpenRouter directly:
hosted runs send none, and the Beaker gateway accepts `reasoning.effort`,
`enabled`, and `exclude` but rejects a token budget. A lower output cap in
`extra_body` is honored; a higher one cannot exceed the cap above. A call whose
reasoning uses the whole cap without answering fails with a model error rather
than ending the rollout as if the agent had finished. The client does not retry
that call, since the same request would stop in the same place, but a hosted run
re-runs the whole case after any model error, up to the run's case retry limit; a
local run scores the world as the agent left it. The cap can still truncate a
legitimate response, so compare task accuracy and truncation when changing it.
Hosted model requests retain per-attempt traces, and returned responses retain
usage/cost accounting.
Tool arguments pass unchanged to PI's MCP tool validation; this client
adds no batch validation or format-recovery instructions. Rebuild the Integration
image from the updated cookbook commit before starting hosted evaluations.

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
BEAKER_AGENT_KEY= uv run python -m pytest -q  # no network, scripted fake client
uv build --no-sources
```

## Attribution

AutomationBench is MIT-licensed by Zapier, Inc. See [ATTRIBUTION.md](ATTRIBUTION.md)
and [LICENSE](LICENSE).
