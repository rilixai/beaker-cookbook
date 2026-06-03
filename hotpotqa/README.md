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

## Sanity-check locally

`rilixai dry-run` builds the spec and runs one case with the seed candidate
— no push, no optimizer loop. Run it from the member directory so it picks up
`[tool.rilixai.spec]`:

```bash
cd hotpotqa
uv run rilixai dry-run --config '{"retrieval_mode": "distractor", "train_size": 1}'
```

It prints the agent output, the per-field scores, and the feedback narratives
the reflection LM would see.

## Run on rilixai

`rilixai_spec.py` registers the `@spec(name="hotpotqa-agent")` runner. The
full build → optimize → pull flow:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/

# Build the image + promote it to hotpotqa-agent@production (run from repo root)
SPEC_VERSION=v$(git rev-parse --short HEAD)
uv run rilixai push --member hotpotqa --version "$SPEC_VERSION"

# Queue a run (reads name/task_type from [tool.rilixai.spec]); tune knobs via --config
cd hotpotqa
uv run rilixai trigger --config '{"max_metric_calls": 6871, "retrieval_mode": "fullwiki", "train_size": 150}'

# Watch it, then download the optimized candidate + per-split reports
uv run rilixai status <run_id>
uv run rilixai pull <run_id> --output-dir hotpotqa_results
```

Defaults are paper-aligned (`max_metric_calls=6871` matches the artifact's
HotpotQA budget). The run config keys are the fields of `HotpotQASandboxConfig`
in `rilixai_spec.py`. `OPENAI_API_KEY` is bound as a project-level secret on
rilixai's side, injected into each sandbox. Roll back with
`uv run rilixai spec promote hotpotqa-agent --version <older-sha>`.

CI (`.github/workflows/push-spec.yml`) runs
`rilixai push --member hotpotqa --version v<short-sha>` on every merge to
`main` that touches `hotpotqa/`: it ships the image and flips `@production`
without spending LLM tokens on a smoke run.

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

## Adapting this recipe to your agent

The whole rilixai integration is `rilixai_spec.py` — one `@spec`-decorated
`HotpotQARunner(BaseCaseRunner)`. To wire your own agent, see the
cookbook's **[`ONBOARDING.md`](../ONBOARDING.md)** for the full contract
(agent expectations, `ComponentApplier` cheatsheet, `FieldConfig` scoring,
feedback, data loading).
