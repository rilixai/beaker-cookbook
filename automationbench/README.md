# AutomationBench + filesystem skills

[AutomationBench](https://github.com/zapier/AutomationBench) ([paper](https://arxiv.org/abs/2604.18934)) is Zapier's benchmark for business automation agents: 600 public tasks in six domains (sales, marketing, operations, support, finance, hr). The agent works in a simulated workspace of SaaS tools (Gmail, Sheets, Slack, CRMs, ...) and is scored by deterministic assertions on the final state of that workspace. Artificial Analysis runs a hosted variant, [AutomationBench-AA](https://artificialanalysis.ai/evaluations/automationbench).

This recipe runs the unmodified benchmark one task at a time (`run_one`) with an agent whose behavior is on disk: a system prompt in `prompts/`, and a `skills/` folder the agent reads through two extra tools, `list_skills` and `read_skill`. Both are read from disk on every call, so Beaker can improve the agent by editing those files between rollouts. The optimizer itself is not part of this recipe.

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

- **Baseline**: `--no-skills` (or no `--skills-dir`). No skill tools; the
  system prompt is `prompts/system_no_skills.md`, which ships as the benchmark's
  own prompt.
- **Skills**: `--skills-dir skills`. The system prompt is `prompts/system.md`
  (the benchmark's prompt plus a paragraph on using the skills) and the agent
  has `list_skills()` / `read_skill(skill_id)` over the skills directory. An
  empty skills directory is fine.

In both arms the task message is the dataset's. `--prompts-dir prompts` selects
the prompt directory; without it the dataset's system prompt is used.

A skill is a folder with a `SKILL.md` (YAML frontmatter `name`/`description`,
markdown body). Its ID is its path under `skills/`:

```text
skills/
  domains/{sales,marketing,operations,support,finance,hr}/SKILL.md
  apps/{gmail,google_sheets,google_drive,slack,salesforce}/SKILL.md
```

The shipped skills are empty (frontmatter only); filling, adding, splitting and
merging them is the optimizer's job. `list_skills` returns every ID with its
description, no filtering. The five seed apps are the most frequent ones in the
tasks' `zapier_tools`.

## Splits

`splits/train.txt` (450 tasks) and `splits/test.txt` (150 tasks) are a fixed
75/25 split per domain, stratified by task family. See
[`splits/README.md`](src/automationbench_skills/splits/README.md) for how they
were made and the leakage policy. The 200-task `simple` domain
(`splits/simple.txt`) is train-only and never scored.

## Models

`--model` defaults to `gpt-5.6-luna` with `--reasoning-effort max`. Routing
follows the benchmark (`vendored/model_setup.py`): `claude-*` goes to
Anthropic, `gemini-*` to the Gemini interactions API, everything else to OpenAI
chat-completions/responses. `--reasoning-effort` maps to each API's reasoning
setting. For a gateway:

```bash
uv run automationbench-skills run --split test --limit 3 \
  --model my-gateway/gemini-2.5-pro --base-url https://gateway.example/v1 --api-key-var GATEWAY_API_KEY
```

`gemini-*` names are routed to Gemini's native API even through a gateway
(upstream behavior). For a plain OpenAI-compatible gateway, use another model
name or pass `--api chat_completions`.

## Reference numbers

Upstream reports strict pass rates (`task_completed_correctly`) of roughly
40–60% for frontier models on the public set. That is not the AutomationBench-AA
number, which adds guardrail and hidden-task components and uses a different
harness.

With this harness on the 150-task test split (`gpt-5-mini`,
`--max-concurrent 16`, one seed):

| arm | pass_rate | partial_credit |
|---|---|---|
| `--no-skills` | 0.013 | 0.221 |
| `--skills-dir skills` (empty seed skills) | 0.053 | 0.294 |

The seed skills are empty, so the gap is mostly run-to-run variance at n=150.
Average a few runs before reading anything into differences this size.

## Beaker integration

```python
from automationbench_skills import load_split, run_one

sample = load_split("train")[0]
result = run_one(sample, skills_dir="skills", prompts_dir="prompts")
result.partial_credit  # 0-1 fraction of assertions passed
result.task_completed_correctly  # strict 0/1
result.trajectory  # full message/tool-call trace
result.end_state  # final simulated WorldState
```

The loop: run train tasks, look at trajectories and scores, edit
`skills/**/SKILL.md` and `prompts/*.md`, run again. Edits are picked up
immediately. Use `test` only for final numbers.

`.beaker/` (`beaker.yaml`, `beaker_spec.py`, `upload_splits.py`) is a working
Beaker integration: it runs a case, scores it with the benchmark's rubric and
traces every model call. Start from it rather than from scratch. The candidate
is `repository=("skills", "prompts")`, i.e. the skill files and the system
prompt.

### How a case is scored

There is no expected answer. Each task is a prompt against a simulated world
(Salesforce, Gmail, Sheets, ... as JSON records). After the agent acts, a list
of assertions is checked against that world
(`salesforce_campaign_member_exists`, `gmail_message_sent_to`, ...); each holds
or does not. Assertions already true before the agent acted don't count, so
doing nothing scores 0.

**Prefer `partial_credit` as the metric to optimize**, the share of assertions
that hold (0–1). It is the objective in `.beaker/beaker_spec.py` and gives a
much denser signal than `task_completed_correctly` (all assertions hold), which
is the strict pass rate: report it, don't optimize for it.

In Beaker's case view each assertion is one check, named with the records it
refers to (`salesforce_campaign_member_exists · David Park · Q1 Product Launch
Webinar`) rather than raw ids. A rollout the model or provider never completed
is a failed case, not a zero.

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run pytest -q          # no network, scripted fake client
```

## Attribution

AutomationBench is MIT-licensed by Zapier, Inc. See [ATTRIBUTION.md](ATTRIBUTION.md)
and [LICENSE](LICENSE).
