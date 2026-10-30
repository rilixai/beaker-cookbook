# Jev BANKING77 support router

A customer message goes to the native TypeSafe SDK as one `choice` question.
Jev selects one of BANKING77's 77 banking support intents and returns its label
probabilities. This is a routing component, not a multi-step autonomous agent.

Beaker optimizes only `prompts/instructions.md` and the intent descriptions in
`prompts/criteria.json`. The intent IDs, application code, dataset preparation,
and exact-match scorer stay fixed. The model is pinned to `jev-1.14.0`.
No LLM judge, chat-completions adapter, or OpenRouter route is used.

## What this benchmark tests

BANKING77 is a fine-grained intent-classification dataset introduced by
[Casanueva et al. (2020)](https://arxiv.org/abs/2003.04807). It contains 13,083
annotated banking support queries across 77 intents. Many intents are closely
related: a message about a card that has not arrived must be distinguished from
one asking for a delivery estimate, for example. The task is to identify the
customer's intended support category from the message's meaning.

This recipe uses Jev as the routing decision-maker. Each message produces one
label and a probability distribution over the fixed intent set. Exact-label
accuracy measures routing correctness; calibration metrics measure how well the
reported probabilities match observed correctness. This tests semantic routing
and confidence quality. It does not measure answer generation, tool execution,
or planning across multiple steps.

The fixed labels and known answers make this a useful Beaker optimization task:
changes to instructions and intent descriptions can be evaluated directly,
without an LLM judge. Training and validation examples support those changes;
the separate official test set measures the frozen candidate's final performance.

## Papers and related work

- **Dataset paper:** Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew
  Henderson, and Ivan Vulić (2020).
  [*Efficient Intent Detection with Dual Sentence Encoders*](https://arxiv.org/abs/2003.04807).
  Introduces BANKING77 and studies intent detection with pretrained sentence
  encoders. This recipe uses its dataset, not its encoder-based classifier.
- **Related Jev evaluation:** Tobias Deußer, Rajkumar Ramamurthy, Lorenz
  Sparrenberg, and Rafet Sifa (2026).
  [*Evaluating and Benchmarking the System One Model Jev*](https://arxiv.org/abs/2609.37647v2).
  Evaluates Jev 1.13 on classification, routing, and other decision tasks, including
  probability calibration. Its results provide context; this recipe uses Jev
  1.14 and its own optimization protocol, so it is not a reproduction of that study.
- **Related open-source benchmark:**
  [JevBench](https://github.com/dhruvmehra/jevbench) compares classifiers on
  BANKING77, AG News, and SST-2, including Jev through OpenRouter. It is a code
  reference rather than a paper, and its reported scores are not this recipe's baseline.

See [ATTRIBUTION.md](ATTRIBUTION.md) for the pinned dataset source, copied
material, and license details.

## Setup and prepare data

From this recipe directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run jev-banking77 prepare --output /tmp/jev-banking77-data
```

Preparation downloads three public files from a pinned PolyAI commit, verifies
their SHA-256 hashes, and writes a provenance manifest. No Hugging Face login is
needed. Source files are cached outside the repository. See [attribution](ATTRIBUTION.md).

| Purpose | Cases | Prepared file |
|---|---:|---|
| Optimization training | 8,463 | `optimization/train.jsonl` |
| Optimization validation | 1,540 (20 per intent) | `optimization/test.jsonl` |
| Final benchmark test | 3,080 | `heldout/test.jsonl` |

The validation split is selected from the 10,003 official training examples by
a fixed hash ordering. Beaker calls its validation file `test.jsonl`; it is
**not the official BANKING77 test set**. Upload only the `optimization/` directory
for optimization. Keep `heldout/` separate, and evaluate it only after freezing
the final candidate. Never choose prompts or thresholds from official test results.

## Run native Jev locally

Set `TYPESAFE_API_KEY` in your shell. The CLI does not read `.env` files. These
commands make paid API calls; start with a small validation sample:

```bash
uv run jev-banking77 evaluate --data /tmp/jev-banking77-data \
  --split validation --limit 20 --output /tmp/jev-baseline.jsonl

# Full validation, using the same fixed rows for baseline and candidate.
uv run jev-banking77 evaluate --data /tmp/jev-banking77-data \
  --split validation --output /tmp/jev-validation.jsonl

# Once the candidate is frozen: all 3,080 official test cases.
uv run jev-banking77 evaluate --data /tmp/jev-banking77-data \
  --split test --output /tmp/jev-heldout.jsonl
```

The CLI writes per-case predictions and a sibling `.summary.json`, including
hashes of the dataset file and routing question used for the run. It reports
accuracy, macro-F1 over all 77 labels, multiclass Brier score, ten-bin expected
calibration error, p50/p95 request latency, and input token use. Requests run
serially. Latency includes SDK retries. Failed requests stay in the accuracy
and F1 denominators and cause a nonzero exit; calibration and latency cover
successful responses only, with their sample count reported separately.

Estimated cost uses the recipe's Jev text-price snapshot of $0.042 per million
input tokens and zero output charge. It is not an invoice, excludes any hidden
retry charges or Beaker fees, and is omitted if any request fails or usage is
missing. Check current TypeSafe pricing before a large run.

There is no measured baseline for this recipe yet. Published scores using Jev
1.13 through OpenRouter are not directly comparable to this pinned native run.

## Use with Beaker

The integration lives under `.beaker/` and uses the same native application
function. It sends only the message to Jev; expected labels are available only
to the deterministic scorer. The objective is mean exact-label accuracy.
Candidate calls record native request content, output, and token usage in traces.

Structural validation with real data (no model call):

```bash
BEAKER_AGENT_KEY= uv run beaker run smoke --strict --integration-id jev-banking77 \
  --config '{"local_dataset_path":"/tmp/jev-banking77-data/optimization"}'
```

Connect your fork and set up a Beaker agent using this recipe's
`.beaker/beaker.yaml`. Upload only the optimization split:

```bash
uv run beaker agent setup "Jev BANKING77" --integration-id jev-banking77
uv run beaker dataset upload /tmp/jev-banking77-data/optimization \
  --name jev-banking77-optimization --split train=8463 --split test=1540
```

Use the immutable dataset revision returned by upload for subsequent smoke and
optimization runs. No organization-specific agent key or dataset ID is committed.
Do not set `scorer_model` or `--optimization-model`; this is a native Jev recipe
with a deterministic scorer, and chat-model overrides are rejected.

`TYPESAFE_API_KEY` is declared in `required_env`. Hosted platform routing can
satisfy it only when the server confirms TypeSafe routing is enabled. That needs
the native gateway from [beaker-ai PR #1570](https://github.com/rilixai/beaker-ai/pull/1570),
a platform key, and the rollout flag. Otherwise provide your own key in encrypted
agent settings. A real customer key bypasses interception and bills directly.
Use a fresh integration build when testing new gateway/runtime code. A structural
smoke pass does not establish live API access, interception, or billing correctness.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run python -m mypy
BEAKER_AGENT_KEY= uv run python -m pytest -q
uv build --no-sources
```

Tests use the real SDK with an in-memory HTTP transport; they make no paid calls.
