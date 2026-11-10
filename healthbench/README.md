# HealthBench

A **health-conversation agent** on OpenAI's
[HealthBench](https://openai.com/index/healthbench/), graded the way the paper
grades it: an LLM judge checks every physician-written rubric criterion, one
call per criterion.

> **Status: WIP.** The harness runs and matches the reference scoring; no
> reference baseline numbers yet.

## The agent

One chat-completions call per conversation (`src/healthbench/agent/agent.py`).

- **Input** — a conversation (1+ turns) between a user and a model, ending in
  a user turn. Users range from worried laypeople to clinicians.
- **Output** — the next assistant turn.
- **The lever** — `agent/prompts.py:SYSTEM_PROMPT`. With one call and no
  tools, the system prompt and the model are what to optimize.

## How responses are scored

Each conversation has its own rubric (~11 criteria on average, 48,562 unique
criteria across the full set).
Criteria carry points from -10 to +10. Negative criteria describe harms (e.g.
"recommends waiting it out at home").

- **Per criterion** — the judge (`evaluation/grader.py`) sees the conversation
  plus the agent's reply and returns `criteria_met: true/false`. The prompt is
  copied verbatim from OpenAI's
  [reference implementation](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py).
- **Per case** — `points met / sum of positive points`. It can go below 0.
- **Overall** — mean over cases, **clipped to [0, 1]**, with a bootstrap std.
- **Breakdowns** — the same score per theme (`theme:emergency_referrals`,
  `theme:context_seeking`, …) and per axis (`axis:accuracy`,
  `axis:completeness`, `axis:communication_quality`, …).

A case whose agent call or judge calls fail scores 0 and is counted in
`num_errored`.

Default judge: `gpt-4.1-2025-04-14`, the grader the paper validated against
physicians. Changing it makes numbers incomparable with published ones.

## The data

Downloaded on first use from OpenAI's public blob store (MIT, via
`openai/simple-evals`) into `~/.cache/healthbench/`.

| `--subset` | Cases | `test` | What it is |
|---|---|---|---|
| `full` (default) | 5,000 | 1,526 | every conversation |
| `hard` | 1,000 | 309 | the hardest for frontier models |
| `consensus` | 3,671 | 1,147 | graded only on the physician-consensus criteria |

### Splits

Derived, not stored (`data/dataset.py`):

- `test` — ~30% of conversations: first byte of `sha256(prompt_id)` < 76.
- `train` — the rest. Optimize here.
- `all` — everything, to compare with published full-set numbers.

Rows in a split are ordered by that hash, so `--limit N` is a uniform random
sample and stable across runs.

## Running it

```bash
cd healthbench
uv sync --group dev
export OPENAI_API_KEY=sk-...
uv run healthbench evaluate --split test --limit 20 --output-dir healthbench_results
```

- `run` — answer the cases, write `run_outputs.json`. No judge calls.
- `evaluate` — reuse any responses in `run_outputs.json`, answer the rest,
  grade everything. Writes `eval_summary.json` and `eval_outputs.json` (every
  criterion's verdict and the judge's explanation).
- `fetch` — download a subset and stop. Needs no key.

Other providers: the OpenAI SDK reads `OPENAI_BASE_URL`, so for OpenRouter set
`OPENAI_BASE_URL=https://openrouter.ai/api/v1`, put your OpenRouter key in
`OPENAI_API_KEY`, and use OpenRouter model names
(`--task-model openai/gpt-4.1-mini --grader-model openai/gpt-4.1`).

### Cost

The judge dominates: ~11 judge calls per case, each carrying the whole
conversation. Start with `--limit 20`.

## Development

```bash
uv sync --group dev --locked
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q   # offline, scripted fake client
```
