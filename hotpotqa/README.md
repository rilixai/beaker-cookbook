# HotpotQA

The HotpotQA multi-hop QA benchmark as a PydanticAI tool-using agent,
optimized by rilixai's GEPA loop.

- **Agent** — a tool-using PydanticAI loop with two tools, terminating on a
  Pydantic `HotpotQAOutput`.
  - `retrieve_k(query)` — deterministic BM25 / fullwiki paragraph retrieval.
  - `summarize(question, passages, context=None)` — raw `AsyncOpenAI` call so
    the optimized prompt is visible at the call site.
- **Optimized prompts** — `policy_prompt` (the agent's `system_prompt`) and
  `summarize_prompt` (the summarize tool's system message).
- **Score** — exact-match + F1 against the gold answer.

Retrieval is pluggable: `fullwiki` (paper parity, bm25s over the 2017
Wikipedia abstracts dump) or `distractor` (HF `hotpot_qa[distractor]`,
10 paragraphs/case — fast and test-friendly).

## Install

```bash
uv sync --all-packages --group dev
```

Env vars (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # agent + summarize (gpt-4.1-mini default)
```

## Run locally

This recipe depends on the lightweight `rilixai` SDK only. The local CLI
covers the two SDK-only paths — `validate` (offline structural check) and
`evaluate` (score one candidate via the SDK `run_case` + scorer loop). The
full GEPA optimize loop runs server-side via `rilixai run` (see the Modal
section below); the optimizer engine lives in the optional `rilixai-runtime`
package, not in this recipe.

```bash
# Validate the spec structure offline (no network, no dataset download)
uv run python -m hotpotqa.cli validate

# Evaluate one candidate (omit --candidate-json to score the seed prompts)
uv run python -m hotpotqa.cli evaluate \
    --split test --candidate-json path/to/candidate.json \
    --output-dir hotpotqa_results/seed
```

Evaluate with no flags scores the seed candidate on the 300-case fullwiki
test slice. See `--help` for all flags.

## Run on Modal (rilixai sandbox)

`optimization/spec.py` registers a `@spec(name="hotpotqa-agent")` factory
that rilixai's sandbox runs. `sandbox.py` builds the image, promotes it to
`hotpotqa-agent@production`, and triggers a run in one shot.

**A dataset upload is required.** The migrated spec no longer loads data
itself — the optimizer reads its cases from an uploaded JSONL dataset via
`HotpotQADataLoader` (row schema: raw `hotpotqa/hotpot_qa` records; see
`HOTPOTQA_DATASET_SCHEMA` in `hotpotqa/data/dataset.py`). A run triggered with
no dataset reference is rejected at startup. Upload a split directory once, then
trigger:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/

# One-time (or when the data changes): upload the JSONL split as a dataset.
uv run rilixai dataset upload --name hotpotqa-agent-dataset path/to/jsonl-dir/

uv run hotpotqa/sandbox.py --build   # build + promote + trigger
uv run hotpotqa/sandbox.py           # trigger only (current @production)
```

The trigger defaults to `--dataset hotpotqa-agent-dataset@production` and
`--spec hotpotqa-agent@production`; override either to pin a specific revision.

`OPENAI_API_KEY` is bound as a project-level secret on rilixai's side,
injected into each sandbox.

To trigger from code, or to tune the agent knobs (`max_metric_calls`,
`retrieval_mode`, models, …), call `client.create_optimization_run(...)` with a
`dataset_ref` — the run config keys are documented in `_DEFAULT_SANDBOX_CONFIG`
at the top of `hotpotqa/optimization/spec.py`. The train/val split is derived
from the uploaded dataset server-side, so there are no `train_size`/`val_size`
knobs. Roll back with `uv run rilixai spec promote hotpotqa-agent v<older-sha>`.

CI (`.github/workflows/push-spec.yml`) runs `sandbox.py --build --no-trigger`
on every merge to `main` that touches `hotpotqa/`: it ships the image and
flips `@production` without spending LLM tokens on a smoke run.

## Tests

```bash
uv run python -m pytest hotpotqa/tests -q
```

Hermetic — scripted PydanticAI `FunctionModel`s, no network.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- The 2017 Wikipedia abstracts dump (~5GB) downloads lazily on first
  `fullwiki` use and caches under `$XDG_CACHE_HOME/rilixai/hotpotqa/fullwiki/`.
- Data slicing is bit-faithful to the GEPA artifact: HotpotQA *train* split
  (90k cases) sliced `[0, 40%)` → test, `[40%, 80%)` → val, `[80%, 100%)` →
  train, sampled with `random.Random(1)`. The 300/300/150 picks match the
  paper.
