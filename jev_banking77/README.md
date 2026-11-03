# Jev BANKING77 support router

Classify banking support messages into 77 intents using native TypeSafe
`choice` requests with `jev-1.14.0`. Beaker optimizes `prompts/instructions.md`
and the intent descriptions in `prompts/criteria.json`. Intent IDs and the
exact-label accuracy scorer stay fixed.

[BANKING77](https://arxiv.org/abs/2003.04807) contains 13,083 labeled queries,
including closely related intents such as a missing card and a request for a
card delivery estimate. Each prediction contains an intent and its label
probabilities; the optimization objective is classification accuracy.

## Prepare data

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run jev-banking77 prepare --output /tmp/jev-banking77-data
```

Preparation downloads checksum-verified files from a pinned upstream commit
and writes a manifest. It selects 20 examples per intent from the official
training set for validation, using a fixed hash ordering.

| Purpose | Cases | Prepared file |
|---|---:|---|
| Optimization training | 8,463 | `optimization/train.jsonl` |
| Optimization validation | 1,540 | `optimization/test.jsonl` |
| Official held-out test | 3,080 | `heldout/test.jsonl` |

Upload only `optimization/` to Beaker. Its `test.jsonl` is the validation split,
not the official test set. Use `heldout/` only after freezing the final candidate.

## Evaluate locally

Set `TYPESAFE_API_KEY` in your shell. The local CLI makes paid API calls and
does not load `.env` files.

```bash
# Small validation run.
uv run jev-banking77 evaluate --data /tmp/jev-banking77-data \
  --split validation --limit 20 --output /tmp/jev-validation.jsonl

# Final candidate on the official test set.
uv run jev-banking77 evaluate --data /tmp/jev-banking77-data \
  --split test --output /tmp/jev-heldout.jsonl
```

The evaluator writes predictions or errors per case and a `.summary.json`
with accuracy, case count, and error count. Failed requests count as incorrect
and cause a nonzero exit. Use `--prompts <directory>` to evaluate other prompt files.

## Run with Beaker

The `.beaker/` integration calls the same application function and records
model-call traces. Expected labels go only to the deterministic scorer.
Hosted calls use Beaker platform routing; no customer TypeSafe key is required.
An existing customer key retains direct provider billing.

```bash
uv run beaker agent setup "Jev BANKING77" --integration-id jev-banking77
uv run beaker dataset upload /tmp/jev-banking77-data/optimization \
  --name banking77 --split train=8463 --split test=1540

# Use the immutable dataset revision returned by upload for both commands.
uv run beaker run smoke --strict --integration-id jev-banking77 \
  --dataset 'banking77@<revision>'
uv run beaker run trigger --integration-id jev-banking77 \
  --dataset 'banking77@<revision>'
```

Smoke validates structure without model calls. Omit `--optimization-model`
and `scorer_model`: this recipe uses native Jev and deterministic scoring.

## References

- Casanueva et al. (2020), [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807), introduces BANKING77.
- Deußer et al. (2026), [*Evaluating and Benchmarking the System One Model Jev*](https://arxiv.org/abs/2609.37647v2), evaluates Jev 1.13. This recipe uses Jev 1.14 and a different optimization protocol.

See [ATTRIBUTION.md](ATTRIBUTION.md) and [DATA_LICENSE](DATA_LICENSE) for the
pinned data source and CC BY 4.0 attribution.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run python -m mypy
BEAKER_AGENT_KEY= uv run python -m pytest -q
uv build --no-sources
```

Tests use an in-memory SDK transport and make no paid calls.
