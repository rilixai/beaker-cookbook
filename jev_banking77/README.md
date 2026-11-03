# Jev BANKING77 support router

Classify 13,083 banking support queries into 77 intents with native TypeSafe
`choice` requests and `jev-1.14.0`. Beaker optimizes `prompts/instructions.md`
and `prompts/criteria.json`; intent IDs and exact-label accuracy scoring stay fixed.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run jev-banking77 prepare --output /tmp/jev-banking77-data
uv run beaker agent setup "Jev BANKING77" --integration-id jev-banking77
uv run beaker dataset upload /tmp/jev-banking77-data/optimization \
  --name banking77 --split train=8463 --split test=1540

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id jev-banking77 \
  --dataset 'banking77@<revision>'
uv run beaker run trigger --integration-id jev-banking77 \
  --dataset 'banking77@<revision>'
```

Preparation verifies pinned source checksums, converts CSV to JSONL, and creates reproducible splits:
8,463 training, 1,540 validation, and 3,080 official test cases. Upload only
`optimization/`: its `test.jsonl` is validation data. Keep `heldout/test.jsonl`
for final evaluation after freezing the candidate.

Hosted runs use platform routing without a customer TypeSafe key. Omit
`--optimization-model` and `scorer_model`. Smoke checks structure without model calls.

## Evaluate locally

Set `TYPESAFE_API_KEY` in your shell; this command makes paid API calls:

```bash
uv run jev-banking77 evaluate --data /tmp/jev-banking77-data \
  --split validation --limit 20 --output /tmp/jev-validation.jsonl
```

For final evaluation, use `--split test`, omit `--limit`, and select candidate
prompts with `--prompts <directory>`. Output includes per-case predictions and
an accuracy summary; request errors count as incorrect and cause a nonzero exit.
Run offline tests with `BEAKER_AGENT_KEY= uv run python -m pytest -q`.

## References

- Casanueva et al. (2020), [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807), introduces BANKING77.
- Deußer et al. (2026), [*Evaluating and Benchmarking the System One Model Jev*](https://arxiv.org/abs/2609.37647v2), evaluates Jev 1.13; this recipe uses Jev 1.14 and a different optimization protocol.

Data and labels: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Intent IDs are unchanged; editable descriptions replace underscores with spaces.
