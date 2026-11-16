# CLINC150 chat router

Route virtual-assistant requests to one of 150 intents, or to `oos` when no
intent fits, with one structured chat-model call per message. Beaker optimizes
`prompts/instructions.md`, `prompts/intents.json` and
`src/clinc150_router/agent.py`; intent IDs, data and exact-label scoring stay fixed.

## How it works

`agent.py` builds a system prompt from the instructions and the 151 intent
descriptions, sends the user's message, and asks for JSON whose `intent` is an
enum of the 151 IDs. An answer outside that set is recorded as an error and
scored as wrong; it is never mapped to `oos`.

Runs started with a model choice call that model through Beaker's inference
gateway. Otherwise Beaker and the local CLI use `gpt-5.6-luna` with `low`
reasoning effort, pinned in `src/clinc150_router/contracts.py`.

Beaker gives each correct `oos` prediction 30 times the weight of a correct
in-scope prediction. On the official validation split, the two groups
contribute equally to the objective. With Beaker's 0–1 per-case scores,
perfect validation predictions yield a mean objective of about 0.0645.
Raw accuracy, in-scope accuracy, and `oos` recall are reported separately.

Every model or decision call must be traced with `trace.model_call`. The chat
call uses operation `chat` with the resolved provider and model. Beaker traces
include the benchmark messages and model replies. A decision
call must use provider `typesafe` with operation `systemone`, or provider
`openai` for the Decisions endpoint, so Beaker can find and compare it.

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

## Evaluate locally

Set `OPENAI_API_KEY` in your shell; this command makes paid API calls:

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

Validation split (3,100 messages), `gpt-5.6-luna`, `low` effort:

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
