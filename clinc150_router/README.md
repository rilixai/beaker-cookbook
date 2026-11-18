# CLINC150 decision router

Route virtual-assistant requests to one of 150 intents, or to `oos` when no
intent fits, with one native TypeSafe decision call per message. Beaker optimizes
`prompts/instructions.md`, `prompts/intents.json` and
`src/clinc150_router/agent.py`; intent IDs, data and exact-label scoring stay fixed.

## How it works

`agent.py` sends the user message as state and asks a single TypeSafe System
One `Choice` question. Its 151 options come from `prompts/intents.json`. The
routing instructions come from `prompts/instructions.md`. The router records an
unknown choice as an error and scores it as incorrect.

The default decision model is `jev-1.14.0`. In Beaker's Decision models comparison,
the gateway swaps the native decision call for the selected target. An ordinary
optimization run uses native Jev and can optimize the routing prompts and code.

Beaker optimizes accuracy. Every correct prediction earns one point, so the
objective is the fraction of cases answered correctly. Beaker also reports
in-scope accuracy and `oos` recall.

The call is traced with provider `typesafe` and operation `systemone`. Traces
include the benchmark messages and decision replies.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run clinc150-router prepare --output /tmp/clinc150-router-data
uv run beaker agent setup "CLINC150 router" --integration-id clinc150-router
uv run beaker dataset upload /tmp/clinc150-router-data/optimization \
  --name clinc150 --split train=15100 --split test=3100

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id clinc150-router \
  --dataset 'clinc150@<revision>'
uv run beaker run trigger --integration-id clinc150-router \
  --dataset 'clinc150@<revision>'
```

Preparation verifies the pinned source checksum and writes three splits, each
sorted by a seeded hash so any prefix is a fair sample:

| File | Upstream split | In-scope | `oos` |
|---|---|---|---|
| `optimization/train.jsonl` | train | 15,000 | 100 |
| `optimization/test.jsonl` | val | 3,000 | 100 |
| `heldout/test.jsonl` | test | 4,500 | 1,000 |

Upload only `optimization/`: its `test.jsonl` is validation data. Keep
`heldout/test.jsonl` for final evaluation after freezing the candidate.
`manifest.json` records the source revision, checksum, seed and counts. Upstream
repeats 3 utterances between train and val and 2 between train and test; they
are kept as published.

To verify decision-model swapping, select the decision-router branch, a small
dataset revision, and **Decision models** in Beaker. Check that the selected
target made a successful decision call on the validation split before scaling
up the comparison.

## Evaluate locally

Set `TYPESAFE_API_KEY` in your shell; this command makes paid API calls:

```bash
uv run clinc150-router evaluate --data /tmp/clinc150-router-data \
  --split validation --limit 200 --output /tmp/clinc150-validation.jsonl
```

For final evaluation, use `--split test`, omit `--limit`, and select candidate
prompts with `--prompts <directory>`. Output includes per-case predictions and a
summary with accuracy, in-scope accuracy, `oos` recall and precision, p50/p95
latency, tokens and cost per 1,000 messages (when cost is available for every
message). Latency includes failed attempts. Request errors count as incorrect
and cause a nonzero exit.
Run offline tests with `BEAKER_AGENT_KEY= uv run python -m pytest -q`.

## Seed baseline

Validation split (3,100 messages), `jev-1.14.0`:

| Metric | Value |
|---|---|
| Accuracy | not yet measured |
| In-scope accuracy | not yet measured |
| `oos` recall | not yet measured |
| `oos` precision | not yet measured |
| Latency p50 / p95 | not yet measured |
| Cost per 1,000 messages | not yet measured |

## References

- Larson et al. (2019), [*An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*](https://aclanthology.org/D19-1131/), introduces CLINC150.

Data and labels: [clinc/oos-eval `data_full.json`](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
Intent IDs are unchanged and out-of-scope is `oos`; editable descriptions replace underscores with spaces.
