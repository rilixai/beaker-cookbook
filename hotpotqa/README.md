# HotpotQA — PydanticAI agent + GEPA optimization

The HotpotQA multi-hop QA task expressed as an idiomatic PydanticAI
tool-using agent, with rilixai's GEPA loop optimizing its two prompts.

* **Tools**
  * `retrieve_k(query)` — deterministic BM25 / fullwiki paragraph retrieval.
  * `summarize(question, passages, context=None)` — LLM-backed summarization
    (raw `AsyncOpenAI` call so the optimized prompt is plainly visible at
    the call site).
* **Optimizable components**
  * `policy_prompt` — the agent's `system_prompt` (tool-use policy).
  * `summarize_prompt` — injected as the `system` message in the summarize
    tool's direct chat-completions call.
* **Terminator** — a Pydantic output type (`HotpotQAOutput.answer`).

Retrieval is pluggable: `fullwiki` (default — paper parity, bm25s over
the 2017 Wikipedia abstracts dump) or `distractor` (the HF
`hotpot_qa[distractor]` 10-paragraph-per-case shape, fast and
test-friendly).

## Install

From the cookbook repo root:

```bash
uv sync --all-packages --group dev
```

(The cookbook is a uv workspace; `--all-packages` installs every
benchmark member.)

Set `OPENAI_API_KEY` (and `ANTHROPIC_API_KEY` if you want to use the
paper's stronger reflection LM) before running anything that calls a
model.

## Run

### Evaluate the seed candidate (paper-faithful test split)

```bash
uv run python -m hotpotqa.cli evaluate \
    --output-dir hotpotqa_results/seed-eval
```

With no flags, evaluates on the 300-case test slice under the paper's
fullwiki + k=7 setup.

### Optimize

```bash
uv run python -m hotpotqa.cli optimize \
    --train-size 150 \
    --max-metric-calls 6871 \
    --reflection-model openai/gpt-4.1 \
    --output-dir hotpotqa_results/optimize-150
```

Defaults are paper-aligned (`--max-metric-calls 6871` matches the
artifact's HotpotQA budget). For full paper parity pass
`--reflection-model openai/gpt-4.1` — the paper uses GPT-4.1 as the
reflection LM (stronger than the GPT-4.1-mini task LM); we leave it
unset by default so the task LM is reused (cheaper, weaker reflection
signal).

After optimize, evaluate the rewritten candidate on the held-out test
split:

```bash
uv run python -m hotpotqa.cli evaluate \
    --candidate-json hotpotqa_results/optimize-150/best_candidate.json \
    --output-dir hotpotqa_results/after-150
```

### Train-size sweep

```bash
uv run python -m hotpotqa.scripts.run_train_size_sweep \
    --output-root hotpotqa_sweep \
    --skip-existing
```

Sweeps a grid of train sizes (default 25, 50, 100, 150, 300), writing
per-point `optimize` + `evaluate` summaries plus a consolidated
`sweep_summary.csv` / `.json`.

## Tests

```bash
uv run python -m pytest hotpotqa/tests -q
```

Tests run hermetically against scripted PydanticAI `FunctionModel`s — no
network access required.

## Notes

* The cookbook installs rilixai from a git+url pin in
  `hotpotqa/pyproject.toml`. When rilixai is published to PyPI at a
  fresh version, the pin swaps to `rilixai==X.Y.Z`.
* Data loading is bit-faithful to the GEPA artifact: source is the
  HotpotQA *train* split (90k cases); fractional slice `[0, 40%)` →
  test, `[40%, 80%)` → val, `[80%, 100%)` → train; sampled with
  `random.Random(1)`. The 300/300/150 picks match the paper.
* The 2017 Wikipedia abstracts dump (~5GB) is downloaded lazily on
  first `fullwiki` use and cached under
  `$XDG_CACHE_HOME/rilixai/hotpotqa/fullwiki/`. First run takes
  several minutes; subsequent runs load from disk in seconds.
