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
Rows hold only `{domain, task_id}`. Expected outcomes come from the pinned upstream tasks plus the fixed reviewed corrections in `src/tau2_bench/task_corrections.json`.

See [the per-sample correction review](docs/dataset-corrections-15f53dd1.md) for the four supported fixes, the local retail-tool fix, and verification limits. These corrections make this a modified benchmark. A new integration build is required; uploading unchanged task-reference rows alone does not apply them. Generated manifests record the corrections file’s SHA-256.

The recipe includes the τ²-bench v2.0.0 Python package in `vendor/tau2` with a
LiteLLM pin that matches the Beaker worker. Both `uv sync` and the Beaker image
build install that copy. For a plain pip install, pass both local projects from
this directory with `python -m pip install . ./vendor/tau2`.

## Models

- **Turn routing:** native TypeSafe System One (`jev-1.14.0`) chooses a reply or one
  of the available tools from the policy and conversation so far.
- **Agent:** without a model choice, `gpt-6-luna` at reasoning `none`, which
  GPT-6 Luna requires for tool calling through Chat Completions. It uses
  `OPENAI_API_KEY` (Beaker's provider proxy in hosted rollouts). A model-comparison run
  sends the selected model through Beaker's inference gateway. `llm.py` calls Chat
  Completions directly with the OpenAI SDK, not LiteLLM. tau2's LiteLLM path sends every
  `gpt-5*` model through the Responses API, and the gateway is OpenAI-compatible Chat
  Completions.
- **User simulator (fixed):** `anthropic/claude-haiku-5-5` at reasoning `low`, chosen
  to reduce simulator cost. It never follows the run's model choice, so a comparison changes
  the agent, not the test. Its calls are traced in `tau2.user_sim` stages.
- **Retail NL-assertion judge (fixed across candidates):** GPT-6 Luna
  (`gpt-6-luna`) at reasoning `low`. It runs in `score_case` through the
  Responses API.

Simulator and judge calls count toward a run's spend, along with the agent's calls.
Traces separate agent cost from simulator cost.

## Scoring

The optimization objective is **`0.7 × partial_correctness + 0.3 × total_correctness`**.
This uses the cookbook's AutomationBench and AppWorld pattern, with a 70/30
weighting: partial progress gets credit, with a bonus for completing the whole task correctly.

- **Partial correctness:** the fraction of scored checks that pass. Database checks
  compare required final state changes, and each required communication fact or
  natural-language assertion is a separate check. Checks outside the task's
  `reward_basis`, including diagnostic reference-action matches, do not count.
- **Total correctness:** 1 only when the conversation ends normally and the original
  tau2 reward is 1; otherwise 0. Existing policy/confirmation assertions still apply;
  the new scorer does not add a universal policy judge.
- **Original benchmark reward:** preserved as both `benchmark_reward` and the existing
  `reward` field, independently of the combined optimization objective.

For database partial credit, the evaluator constructs the initial world, applies the
reference actions to get the expected final world, and replays the agent trajectory
in another world. Each changed record field is a check. Lists of records, such as
ordered items, are checked by position per entry; other lists and nested fields
(such as an address) are atomic. New/deleted records are atomic too. These are
mechanically derived state checks, not a hand-authored semantic task rubric.
Equivalent tool sequences that reach the same final state receive the same credit.
Untouched data earns no credit, and unexpected changes add failed checks. Comparison
units come from the initial/reference states, so truncating a predicted list does
not remove failed requirements. A task requiring no database changes has no database
partial-credit checks; its required conversation checks determine partial credit.
If there are no scored checks at all, partial credit falls back to the original
reward. Reference reads do not change state and are skipped for partial-credit
replay. If a reference write fails, the scorer falls back to the original binary
database check for that task and reports an informational `REFERENCE` failure.
It does not treat a partially replayed reference as a valid set of outcomes.
Other replay errors still surface as evaluation errors.

For example, passing 3 of 4 scored requirements with original reward 0 gives
`partial_correctness=0.75`, `total_correctness=0`, and objective `0.525`.
Passing everything gives objective `1.0`.

The original tau2 reward remains the product of the task's required components:

| Domain | Original reward | Deterministic? |
|---|---|---|
| airline | `DB` × `COMMUNICATE` | Yes. Communicate checks use the **substring** scorer (`--communicate-scorer substring`). |
| retail | `DB` × `NL_ASSERTION` (2 tasks: `DB` only) | No. GPT-6 Luna grades the NL assertions. |

`field_scores` expose `partial_correctness`, `total_correctness`,
`combined_correctness`, `benchmark_reward`, the legacy `reward`, and the original
component scores (`db`, `communicate`, `nl_assertion`). `checks` expose database
outcomes with expected/actual values and the existing conversation checks. The
whole-database match remains visible as an informational summary when detailed
outcomes are available; it does not double-count database correctness.

This is a custom optimization metric, not the published tau2 reward. Compare
historical runs using `benchmark_reward`, or rescore recorded trajectories with the
same scorer version. Contradictory customer instructions/reference outcomes still
need data fixes; partial credit does not repair them.

The evaluator runs in `score_case` on the recorded trajectory. It replays the tool
calls to rebuild the final database. The database itself is not stored in the result,
because the airline and retail databases are larger than Beaker's result limit.

These cases score 0:

- A conversation that ends early (turn cap, too many errors, agent protocol error).
  Every component is reported as 0.
- An exception from the agent or simulator. The case fails and shows the error.

Provider outages (connection errors, rate limits, 5xx responses) are retried, not scored.

## Optimizing the decision model

The seed uses the decision on every turn with tools available. When Jev selects
a tool, the chat model receives only that tool and must generate its arguments.
When Jev selects a reply, the chat model receives no tools. The chat model still
handles wording and argument generation. `llm.py` contains the editable decision
instructions and choices. The baseline records these calls as provider
`typesafe`, operation `systemone`, with token usage.

Run the normal Beaker optimization on this branch with TypeSafe provider routing.
A Beaker chat model override changes the chat model only. Use the existing
dataset and score to compare changes to the decision instructions or routing.

## Evaluate locally

Set `OPENAI_API_KEY` and `TYPESAFE_API_KEY`. This command makes paid calls for the agent, the simulator and
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
