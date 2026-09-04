# AutomationBench + filesystem skills

[AutomationBench](https://github.com/zapier/AutomationBench) ([paper](https://arxiv.org/abs/2604.18934)) is Zapier's benchmark for agentic business automation: 600 public tasks across six domains (sales, marketing, operations, support, finance, hr) where an agent works in a simulated workspace of SaaS tools (Gmail, Sheets, Slack, CRMs, ...) and is scored by deterministic assertions on the final world state. A hosted variant, [AutomationBench-AA](https://artificialanalysis.ai/evaluations/automationbench), is run by Artificial Analysis.

This recipe wraps the unmodified benchmark in a per-sample runner (`run_one`) plus a filesystem `skills/` hook: two extra tools (`list_skills`, `read_skill`) that read `SKILL.md` files live from disk on every call. That makes the harness directly drivable by **Beaker** — an optimizer that improves the agent purely by editing the `skills/` folder between rollouts. This recipe is the harness + baseline + skills hook only; it does not implement the optimizer.

## Quick start

```bash
cd automationbench
uv sync --group dev
export OPENAI_API_KEY=sk-...   # or copy .env.example to .env

# Smoke run: 3 test tasks, with the seed skill stubs
uv run automationbench-skills run --split test --limit 3 --model gpt-5-mini --skills-dir skills

# Aggregate a finished (or partial) run directory
uv run automationbench-skills evaluate --output-dir runs/<run-dir>
```

`run` writes one JSON per task (both scores + full trajectory + final world state)
plus `config.json` and `summary.json` into `--output-dir` (default
`runs/<split>-<timestamp>`). `evaluate` prints pass rate (mean
`task_completed_correctly`) and mean `partial_credit`, per domain and overall.
`--task-timeout <seconds>` bounds each rollout (a stuck API request otherwise
hangs the run); timed-out tasks score 0 with `error="timeout ..."`.

## Baseline vs. skills

- **Baseline**: `--no-skills` (or omit `--skills-dir`) — the skill tools are not
  registered at all; this is the stock benchmark agent.
- **Skills arm**: `--skills-dir skills` — the agent gets `list_skills()` /
  `read_skill(skill_id)` over that directory. Files are re-read on every tool
  call, so editing them between rollouts changes behavior with no restart. An
  empty directory is valid (tools present, no skills).

Skills are folders on two axes, each holding a `SKILL.md` (YAML frontmatter
`name`/`description` + markdown body, i.e. an Anthropic Agent Skill). A skill's
ID is its path under `skills/`:

```text
skills/
  domains/{sales,marketing,operations,support,finance,hr}/SKILL.md
  apps/{gmail,google_sheets,google_drive,slack,salesforce}/SKILL.md
```

The shipped stubs are intentionally empty (frontmatter only):
`list_skills` returns every ID with its description — no domain
filtering — and the optimizer is expected to fill, add, split, and merge skills
across both axes. The five seed apps are the highest-frequency apps across the
tasks' `zapier_tools`.

The benchmark's own domain system prompts are never modified; the only nudge to
consult skills lives in the tools' own descriptions.

## Splits

`splits/train.txt` (450 tasks) and `splits/test.txt` (150 tasks) freeze a
75/25-per-domain partition of the 600 scored public tasks, stratified across
task families with a fixed seed — see [`splits/README.md`](src/automationbench_skills/splits/README.md)
for the exact procedure, regeneration command, and leakage policy. The optional
200-task `simple` domain is provided as train-only material (`splits/simple.txt`)
and is never scored.

## Models

`--model` defaults to `gpt-5-mini`. Routing reuses the benchmark's own logic
(adapted in `vendored/model_setup.py`): `claude-*` → Anthropic native,
`gemini-*` → Gemini interactions API, everything else → OpenAI
chat-completions/responses. `--reasoning-effort` maps to each API's native
reasoning knob. Gateway models work via an OpenAI-compatible endpoint:

```bash
uv run automationbench-skills run --split test --limit 3 \
  --model my-gateway/gemini-2.5-pro --base-url https://gateway.example/v1 --api-key-var GATEWAY_API_KEY
```

Note: routing gateways to Gemini's native interactions API is upstream behavior
for `gemini-*`-named models; plain OpenAI-compatible gateways should use
non-Gemini model names or `--api chat_completions`.

## Reference numbers

Upstream reports public-set strict pass rates (`task_completed_correctly`) of
roughly 40–60% for frontier models (see the upstream README's leaderboard).
Caveat: the public strict pass rate is **not** the AutomationBench-AA number —
AA's headline metric includes guardrail/hidden-task components and a different
harness, so scores are not directly comparable.

Reference runs with this harness on the frozen 150-task test split
(`gpt-5-mini`, `--max-concurrent 16`, single seed):

| arm | pass_rate | partial_credit |
|---|---|---|
| `--no-skills` baseline | 0.013 | 0.221 |
| `--skills-dir skills` (empty seed stubs) | 0.053 | 0.294 |

The stubs are frontmatter-only, so the gap is likely dominated by run-to-run
variance at n=150 rather than the empty skills helping — average several runs
before reading much into arm differences of this size.

## Beaker integration

```python
from automationbench_skills import load_split, run_one

sample = load_split("train")[0]
result = run_one(sample, model="gpt-5-mini", skills_dir="skills")
result.partial_credit  # 0-1 fraction of assertions passed
result.task_completed_correctly  # strict 0/1
result.trajectory  # full message/tool-call trace
result.end_state  # final simulated WorldState
```

An optimizer loop: run train samples, inspect trajectories/scores, edit
`skills/**/SKILL.md`, rerun — the environment is reused across calls and picks up
skill edits immediately. Evaluate on `test` only for final reporting.

The Beaker integration is pre-configured in `.beaker/` (`beaker.yaml`,
`beaker_spec.py`, `upload_splits.py`); nothing to wire up.

### How a case is scored

There is no expected answer. A task is a prompt against a simulated world
(Salesforce, Gmail, Sheets, ... as JSON records); after the agent acts, a list
of deterministic assertions is checked against that world
(`salesforce_campaign_member_exists`, `gmail_message_sent_to`, ...). Each one
holds or does not. Assertions already true before the agent acted are excluded,
so doing nothing scores 0.

**Prefer `partial_credit` as the metric to optimize** (share of assertions that
hold, 0–1): it is the default objective in `.beaker/beaker_spec.py` and gives a
denser signal per task than `task_completed_correctly` (all hold), the strict
pass rate, which is still worth reporting.

In Beaker's case view each assertion is one check, named with the records it
refers to (`salesforce_campaign_member_exists · David Park · Q1 Product Launch
Webinar`) instead of raw ids; display only. A rollout the model or provider
never completed is a failed case, not a zero.

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run pytest -q          # hermetic: no network, scripted fake client
```

## Attribution

AutomationBench is MIT-licensed by Zapier, Inc. — see [ATTRIBUTION.md](ATTRIBUTION.md)
and [LICENSE](LICENSE).
