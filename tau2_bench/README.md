# τ²-bench customer-service agent

A tool-calling customer-service agent on [τ²-bench](https://github.com/sierra-research/tau2-bench)
v2.0.0, for the **airline** and **retail** domains (English, text). Each case is one
τ²-bench task, run as a full conversation with upstream's LLM user simulator and
scored by upstream's evaluator.

Beaker may edit `prompts/system.md`, the agent loop (`src/tau2_bench/agent.py`) and its
chat-call layer (`src/tau2_bench/llm.py`). Task data, domain policies, the user
simulator, the orchestrator and scoring stay fixed.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run tau2-bench prepare --output /tmp/tau2-bench-data
uv run beaker agent setup "tau2-bench" --integration-id tau2-bench
uv run beaker dataset upload /tmp/tau2-bench-data/optimization \
  --name tau2 --split train=77 --split test=27

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id tau2-bench --dataset 'tau2@<revision>'
uv run beaker run trigger --integration-id tau2-bench --dataset 'tau2@<revision>'
```

`prepare` downloads the 12 upstream data files an English airline/retail run reads
(about 42 MB), pinned to the release commit and checked against SHA-256 digests, into
`~/.cache/tau2-bench/<commit>/data` (override with `TAU2_DATA_DIR`). Hosted rollouts
fetch the same files on first use.

| Split | Airline | Retail | File |
|---|---|---|---|
| train | 22 | 55 | `optimization/train.jsonl` |
| validation | 8 | 19 | `optimization/test.jsonl` (uploaded as Beaker's test split) |
| held-out test | 20 | 40 | `heldout/test.jsonl` |

Upstream's official `test` split is the held-out test. Upstream's `train` split is
divided 75/25 into train and validation by a hash with seed `tau2-bench-v1`. Upload
only `optimization/`, and keep `heldout/` for the final evaluation of a frozen candidate.
Rows hold only `{domain, task_id}`. Expected outcomes stay inside upstream.

## Models

- **Agent:** without a model choice, `gpt-5.6-luna` at reasoning `medium`, through
  `OPENAI_API_KEY` (Beaker's provider proxy in hosted rollouts). A model-comparison run
  sends the selected model through Beaker's inference gateway. `llm.py` calls Chat
  Completions directly with the OpenAI SDK, not LiteLLM. tau2's LiteLLM path sends every
  `gpt-5*` model through the Responses API, and the gateway is OpenAI-compatible Chat
  Completions.
- **User simulator (fixed):** `openai/gpt-5.5` at reasoning `low`, upstream's 2.0
  benchmark default. It never follows the run's model choice, so a comparison changes
  the agent, not the test. Its calls are traced in `tau2.user_sim` stages.
- **Retail NL-assertion judge (fixed):** upstream's `gpt-5.4-mini-2026-03-17` at
  reasoning `low`. It runs in `score_case` with `OPENAI_API_KEY`.

Simulator and judge calls count toward a run's spend, along with the agent's calls.
Traces separate agent cost from simulator cost.

## Scoring

The objective is τ²-bench's reward, 0 or 1: the product of the task's `reward_basis` components.

| Domain | Reward | Deterministic? |
|---|---|---|
| airline | `DB` × `COMMUNICATE` | Yes. Communicate checks use the **substring** scorer (`--communicate-scorer substring`), not the default LLM judge. |
| retail | `DB` × `NL_ASSERTION` (2 tasks: `DB` only) | No. NL assertions are judged by upstream's pinned LLM judge. |

`field_scores` hold `reward`, `db`, plus `communicate` (airline) or `nl_assertion`
(retail). `checks` show how the conversation ended, the DB match, and one row per
communicate string or NL assertion. Upstream's gold-action matches also appear, marked
informational, because they are not part of either domain's reward.

The evaluator runs in `score_case` on the recorded trajectory. It replays the tool
calls to rebuild the final database. The database itself is not stored in the result,
because the airline and retail databases are larger than Beaker's result limit.

These cases score 0:

- A conversation that ends early (turn cap, too many errors, agent protocol error).
  Every component is reported as 0.
- An exception from the agent or simulator. The case fails and shows the error.

Provider outages (connection errors, rate limits, 5xx responses) are retried, not scored.

## Adding decision calls

The seed agent makes no decision-API calls. If you add one, for example a closed-set
policy judgment beside the chat call in `llm.py` or `agent.py`, trace it as a model call.
Use provider `typesafe` with operation `systemone`, or provider `openai` with the
decisions operation, as in [`jev_banking77/.beaker/beaker_integration.py`](../jev_banking77/.beaker/beaker_integration.py).
Untraced decision calls cannot be found in baseline traces.

## Evaluate locally

Set `OPENAI_API_KEY`. This command makes paid calls for the agent, the simulator and
(retail) the judge:

```bash
uv run tau2-bench evaluate --data /tmp/tau2-bench-data \
  --split validation --limit 5 --output /tmp/tau2-validation.jsonl
```

Use `--domain airline|retail` to run one domain and `--prompts <dir>` to select candidate
instructions. The output has per-case rewards, components and conversation summaries.
A `.summary.json` file holds the mean reward per domain and the mean agent turns. Errors
count as failed cases and cause a nonzero exit.

Run the offline tests with `BEAKER_AGENT_KEY= uv run python -m pytest -q`. The tests
stub every model call. On first use they download the pinned public data.

## Baseline, noise and cost

**Not measured yet.** These numbers need paid runs: run the seed on validation 3 times
and record each domain's mean reward and spread, and the mean turns, tokens and cost per case.

The user simulator is stochastic, so a single case can flip between runs. If noise
hides the differences between models, a later version can add upstream's `solo_mode`
(no simulator) as a second integration.

## References

- Barres et al. (2025), [*τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment*](https://arxiv.org/abs/2506.07982).
- Upstream: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) v2.0.0,
  MIT. See [ATTRIBUTION.md](ATTRIBUTION.md).
