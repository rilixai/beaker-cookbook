# LiveNewsBench on the You.com Inference Router

[LiveNewsBench](https://github.com/YunfanZhang42/LiveNewsBench) ([paper](https://arxiv.org/abs/2602.13543))
asks hard questions about recent news events. In
[Braintrust's August 2026 study](https://www.braintrust.dev/evals/search-relevance-eval),
models scored 3–51% without web search and 79–85% with You.com search. Answers are
short and graded by an LLM judge against a gold answer.

The agent here is a tool-calling loop. Its model is an open-weight model on the
[You.com Inference Router](https://you.com/product/inference-router), and its two
tools use You.com's [Search](https://you.com/docs/guides/search) and
[Contents](https://you.com/docs/guides/contents) APIs. Beaker edits the agent
(`src/livenewsbench/agent/`). The model, the web rules and the grader are fixed.

> **The Beaker integration is already present in this recipe.** `.beaker/` holds a
> working `beaker.yaml` and `beaker_integration.py`.

## Rules

These follow upstream's reference agent and are enforced in `src/livenewsbench/tools.py`, outside the editable agent:

- At most **5 searches and 5 page visits** per question, upstream's default. A call over budget is refused before any request is sent.
- A page can be opened only if a search in the same question returned it.
- Search excludes Internet archive sites (upstream's rule) and the hosts that publish this benchmark's questions with answers: Hugging Face, GitHub, arXiv, Braintrust. `BLOCKED_DOMAINS` lists them. Results from these hosts are also dropped after the search.
- At most **12 model calls** per question (`MAX_MODEL_CALLS`). The seed agent keeps the last one for its answer.
- A search query may have at most 50 words and 400 characters, You.com Search's limit. A longer one is refused before it is sent and uses no search.
- If You.com rejects a search or page itself (an HTTP 4xx other than a key error), the agent is told and the search or page visit counts. The question goes on.

The model is `deepseek-v4.1-flash` by default. Set `LIVENEWSBENCH_MODEL` to another
[router model ID](https://you.com/product/inference-router/models), such as `glm-5.3`
or `kimi-k3`. Sampling settings are left to the router, which applies each vendor's
reference values. `max_tokens` is 16,384, because reasoning tokens count against it.

The grader is upstream's: the SimpleQA grading prompt sent to `gpt-4.1` at
temperature 0. An answer is correct only if the grader replies `A`. A reply other
than A, B or C is retried twice, then the case fails with an error. An empty answer
is graded NOT_ATTEMPTED without a grader call.

## Splits

The pinned release is `jan_2026_release_2`. It has train 612 (May–Sep 2025 events),
val 171 (Oct 2025), test 346 (Nov–Dec 2025) and human-verified test 200.

The 200 human-verified questions are a **subset of test**. The release therefore has
1,129 unique questions, not the 1,329 the split sizes add up to.

| Use | Split | Cases |
|---|---|---|
| Beaker train | upstream train | 612 |
| Beaker test | upstream val | 171 |
| Final report, after freezing a candidate | upstream test, and its human-verified subset | 346 / 200 |

`prepare` writes `optimization/` for upload and keeps `heldout/test.jsonl` out of Beaker.

Every hypothesis re-runs the whole train split, so its size sets how long a run takes.
`prepare --train-sample 300` keeps 300 train questions. Each month-and-category group keeps
its share, and `--seed` (default 0) fixes which questions are kept. The manifest records
the size and seed. Test stays at 171.

Scores repeat closely between runs: in our first run, 2 of 95 questions changed when run
twice. So 300 questions still show gains of about 1 point, while every evaluation takes
half the time. Don't pick the subset from the questions the baseline failed: some of
those failures are noise, and they "improve" on their own when run again.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run livenewsbench prepare --output data
uv run beaker agent setup "LiveNewsBench" --integration-id livenewsbench
uv run beaker agent env set YDC_API_KEY --value-stdin   # paste a You.com API key
uv run beaker dataset upload data/optimization \
  --name livenewsbench-jan2026 --split train=612 --split test=171

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id livenewsbench --dataset 'livenewsbench-jan2026@<revision>'
uv run beaker run trigger --integration-id livenewsbench --dataset 'livenewsbench-jan2026@<revision>'
```

Omit `--optimization-model`, so the agent runs on the You.com router. Passing one
sends the agent's model calls through Beaker's gateway instead; search still uses
You.com. That gives a comparison run on the same harness. The grader runs through
Beaker's gateway as `scorer_model: openai:gpt-4.1`.

**Cost tracking.** Router, Search and Contents calls go straight to You.com, so
Beaker's run spend does not include them. The trace still records every model call's
tokens, and the search and page-visit counts. Use `evaluate` below for dollar
figures.

**Rate limits.** The Search and Contents APIs default to 10 requests per second. The
client retries their 429s with `Retry-After` and backs off on 5xx.

The router throttles too. Our first hosted run used Beaker's default of 32 cases at
once: cases finished at a flat ~25 a minute, and the median rollout took 44 s, against
6 s in a local run at 8 at once. When the OpenAI client's own retries end on a router
429, the agent's model call waits up to four more times, about 3 minutes in all
(`RATE_LIMIT_WAITS`). Each case's usage records these waits as `rate_limit_waits` and
`rate_limit_seconds`.

A provider failure that outlasts all retries raises `RetryableCaseError`. That marks the
case as worth retrying instead of scoring it 0. Ask You.com for your key's limits, and
set `--config '{"max_concurrency": 8}'` on `run trigger`, so that latency measures the
router, not its rate limit.

## Baseline: GPT with OpenAI's built-in web search

The `livenewsbench-openai` integration runs `gpt-6-luna` with OpenAI's built-in `web_search` tool
on the Responses API. It uses the same data and grader, so it can be compared with runs on the
You.com router and search. Its editable agent is a separate package,
`src/livenewsbench/openai_agent/`. The fixed harness is `openai_search.py`.

```bash
uv run beaker agent setup "LiveNewsBench OpenAI" --integration-id livenewsbench-openai
uv run beaker dataset upload data/optimization --agent livenewsbench-openai \
  --name livenewsbench-jan2026 --split train=612 --split test=171
uv run beaker run trigger --integration-id livenewsbench-openai --dataset 'livenewsbench-jan2026@<revision>'
```

- **Leave out `--optimization-model`.** The agent calls `api.openai.com` itself. Beaker's
  provider proxy routes those calls through Beaker's gateway, which meters tokens and
  searches; the gateway accepts built-in web search only on that route.
- **Don't set `OPENAI_API_KEY` on this agent.** A key set there sends the calls straight to
  OpenAI, outside Beaker's metering.
- **To change the model,** set `LIVENEWSBENCH_MODEL` on the agent.

OpenAI runs the search loop inside each request, so two rules work differently:
- **Budget.** Searches, page opens and in-page finds share one budget of 10 per question,
  the You.com side's 5 searches plus 5 page visits. Each request sets `max_tool_calls` to
  what is left.
- **Blocked hosts.** OpenAI's `filters` take only an allowlist, so the blocked hosts can't
  be excluded up front. Every URL the model searched, opened or cited is checked afterwards.
  - An answer from a question that opened, searched in or cited a blocked host is withheld,
    so it scores as not attempted. `evaluate` reports these as `withheld_blocked_host`.
  - Blocked hosts that only appeared among search results are counted in `blocked_in_results`.

Reasoning effort is pinned to `medium`, the model's default.

## Evaluate locally

`evaluate` makes paid API calls. It needs `YDC_API_KEY` (except with `--search openai`), and `OPENAI_API_KEY` for the grader.
It stops before any rollout if OpenAI rejects the grader's key.

```bash
uv run livenewsbench evaluate --data data --split val --limit 20 --output runs/val.jsonl
```

For the final report, use `--split test` or `--split human_verified_test`, without
`--limit`. Pass `--agent <dir>` to evaluate a candidate's `src/livenewsbench/agent`.

To compare another endpoint on the same harness and You.com search, pass
`--base-url`, `--api-key-env` and `--model`. For example, a frontier model:
`--base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model gpt-6.1-sol`.

For the OpenAI baseline, pass `--search openai`. It uses `OPENAI_API_KEY` and doesn't need
`YDC_API_KEY`; the model defaults to `gpt-6-luna`. Its cost uses OpenAI's list prices:
`gpt-6-luna` tokens at the rates pinned in `openai_search.py`, and $10 per 1,000 searches.
Page opens aren't billed.

The command writes one JSON line per question: answer, grade, model calls,
input/cached/output/reasoning tokens, searches, page visits, seconds, and an
estimated cost. A failed case records only the exception type and `error_stage`
(`rollout` or `grading`). When grading fails, the answer, usage and cost are kept.
The `.summary.json` reports:
- accuracy, with errors counted as wrong
- seconds per question: mean, p50, p90
- mean searches, page visits and model calls
- cost per question and per correct answer

Cost is You.com's list prices applied to the measured usage:
- router tokens at the [models page](https://you.com/product/inference-router/models) rates, pinned in `youcom.py`
- $5 per 1,000 searches
- $1 per 1,000 pages

The model share is reported only for router models.

## Differences from upstream's reference agent

- Native function tools (`search`, `open_page`) instead of text `<action>` blocks.
- You.com Search instead of Tavily. Pages are fetched on demand through the Contents
  API, where Tavily returns them with the search.
- The seed agent cuts pages at 24,000 characters, roughly 6k tokens; upstream cuts at
  8k tokens. Beaker may change this.
- More hosts are blocked (listed above). The jan_2026 val and test rows carry no
  source links, so per-question source-domain exclusion is not possible there; the
  reference agent does not apply it either.

The router's server-side `web_search`/`web_extract` tools are not used. One
request can run several searches per round, so they cannot enforce 5 searches and
5 page visits exactly, and `web_extract` takes no domain exclusions.

## Checks

```bash
uv run ruff check && uv run ruff format --check
uv run python -m mypy
uv run python -m pytest -q
```

The tests are offline: You.com and OpenAI are mock transports.
