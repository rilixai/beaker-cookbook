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

## Run on Modal (rilixai sandbox)

The cookbook has two entry points:

| Script | Where it runs | When to use |
|---|---|---|
| `hotpotqa/cli.py`     | Your laptop                 | Local debugging, evaluating a downloaded candidate, no rilixai account needed |
| `hotpotqa/sandbox.py` | rilixai hosted Modal sandbox | Real optimization runs, sweeps, scheduled retraining, sharing candidates across teams |

(A follow-up PR will unify these into one subcommand-style CLI.)

`hotpotqa/optimization/spec.py` registers a `@spec(name="hotpotqa-agent")`
factory that rilixai's sandbox dispatcher invokes once per run. No
version is pinned in the decorator — `sandbox.py --build` supplies
`v<short_sha>` at push time and **promotes the freshly-pushed row to
`hotpotqa-agent@production`** in the same invocation. Triggers
reference `hotpotqa-agent@production`, which rilixai resolves
server-side to whatever's currently promoted.

This mirrors the [rilix prod-latest pattern](https://github.com/rilixai/rilix/pull/1381)
— no manual version bumps in source for routine deploys, and rollback
is `uv run rilixai spec promote hotpotqa-agent --version <older>`.

### Prerequisites

```bash
export RILIXAI_API_KEY=sk-...                                            # rilixai control-plane credential
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/   # CDK stack ApiUrl output
```

The `OPENAI_API_KEY` the agent + summarize tool need is bound at the
*project* level on rilixai's side, not on your machine — see rilixai's
docs for project-secret configuration. Once bound, every run spawned
under that project receives it as an env var inside the sandbox.

**Scope.** Every optimization run belongs to a `scope_key` — a
customer-defined stable string (e.g. `"hotpotqa-agent"`) that groups
runs + releases. The scope row is auto-created on first reference, so
just pick a value and pass it to every `create_optimization_run` call.

### One-shot: build + promote + trigger via `sandbox.py`

The canonical CI-equivalent flow:

```bash
uv run hotpotqa/sandbox.py --build
```

What this does:

1. **Push** the cookbook source as a fresh Modal image, registered as
   `hotpotqa-agent@v<short_sha>` (the short SHA comes from
   `git rev-parse --short HEAD` automatically).
2. **Promote** the just-pushed version: `hotpotqa-agent@production`
   now resolves to it.
3. **Trigger** a run referencing `hotpotqa-agent@production`.

For trigger-only flows (no rebuild needed — pick up whatever's
currently `@production`):

```bash
uv run hotpotqa/sandbox.py
```

To pin a specific build for smoke / regression debugging without
flipping production:

```bash
# Push a fresh build but don't promote it:
uv run hotpotqa/sandbox.py --build --no-promote

# Then trigger against that specific version:
uv run hotpotqa/sandbox.py --spec hotpotqa-agent@v1a2b3c4
```

### Phase 1 — push the spec (image build)

`rilixai push` bundles the cookbook source tree, ships it to rilixai's
build worker, which installs the `hotpotqa` workspace member's deps
into a Modal image and registers `hotpotqa-agent@v<short_sha>` against
your project. Re-run only when code or deps change.

```bash
cd ~/Projects/rilixai-cookbook
SPEC_VERSION="v$(git rev-parse --short HEAD)"
uv run rilixai push \
    --source-dir . \
    --name hotpotqa-agent \
    --version "$SPEC_VERSION" \
    --pip-install "pydantic-ai>=0.0.20" \
    --pip-install "rank-bm25>=0.2" \
    --pip-install "datasets>=2.14" \
    --pip-install "bm25s>=0.2" \
    --pip-install "PyStemmer>=2.2" \
    --pip-install "ujson>=5" \
    --pip-install "huggingface-hub>=0.20" \
    hotpotqa/optimization/spec.py

# Then promote to production:
uv run rilixai spec promote hotpotqa-agent --version "$SPEC_VERSION"
```

The `--pip-install` entries are required because the build worker runs
`pip install /spec` against the **cookbook root** `pyproject.toml`
(which declares zero deps — it's just a uv workspace marker). The
workspace member's deps live in `hotpotqa/pyproject.toml` and are
invisible to the build worker, so they have to be passed explicitly.
`sandbox.py --build` does this for you by reading the member pyproject
at runtime.

(The spec-file path is positional, not a flag — see `rilixai push --help`.)

What happens server-side:

1. Source tree zipped + uploaded to a presigned S3 URL.
2. Build worker bakes a Modal image: base Python 3.12 + `pip install
   ./hotpotqa` (pulls rilixai via the git+url pin + bm25s + datasets
   + pydantic-ai + ...).
3. `SpecVersion` row stamped with `status=READY` and `image_ref=<modal
   image id>`. The CLI polls until READY and exits.
4. `spec promote` flips `hotpotqa-agent@production` to point at the
   just-pushed row.

First-time builds take ~3–8 minutes. Subsequent pushes with cached
layers finish in under a minute. Older versions stay addressable
forever (`spec="hotpotqa-agent@v<older-sha>"`) so historical runs are
reproducible; rollback is `uv run rilixai spec promote hotpotqa-agent
--version <older>`.

### Phase 2 — trigger a run

```python
from rilixai import RilixAIClient

client = RilixAIClient(
    base_url="https://api.rilix.ai",
    api_key="sk-...",     # your RILIXAI_API_KEY
)

response = client.create_optimization_run(
    task_type="hotpotqa_pydantic_agent",
    spec="hotpotqa-agent@production",  # resolves to the currently promoted version
    scope_key="hotpotqa-agent",          # customer-defined stable key; auto-created on first use
    config={
        # ─── GEPA per-run knobs (consumed by rilixai's sandbox runtime) ─────
        "max_metric_calls": 2000,             # ⚠ GEPA budget — primary cost knob
        "reflection_minibatch_size": 3,
        "reflection_model": "openai/gpt-4.1", # paper-parity reflection LM
        "seed": 0,

        # ─── HotpotQA cookbook knobs (consumed by build_spec in spec.py) ───
        "retrieval_mode": "distractor",       # default; "fullwiki" for paper parity
        "retrieve_k": 7,
        "max_iters": 8,
        "train_size": 50,                     # paper uses 150
        "val_size": 100,                      # paper uses 300
        "pydantic_agent_model": "openai:gpt-4.1-mini",
        "task_temperature": 0.0,
    },
)
run_id = response["id"]
print(f"queued run: {run_id}")
```

**Where `max_metric_calls` is set** (the question every cost-conscious
caller asks first): inside the trigger's `config={…}` dict, as a
top-level key. rilixai's sandbox runtime parses it directly into the
GEPA `PromptOptimizationRunConfig` — **`build_spec` does not see or
control it**. The split between "GEPA knobs" and "HotpotQA cookbook
knobs" in the config above mirrors that boundary:

| Key                          | Owned by               | Default if omitted                  |
|------------------------------|------------------------|-------------------------------------|
| `max_metric_calls`           | rilixai sandbox        | 50 (rilixai default — bump it!)     |
| `reflection_minibatch_size`  | rilixai sandbox        | 3                                   |
| `reflection_model`           | rilixai sandbox        | (reuses task model)                 |
| `seed`                       | rilixai sandbox        | 0                                   |
| `retrieval_mode`             | hotpotqa `build_spec`  | `"distractor"`                      |
| `retrieve_k`                 | hotpotqa `build_spec`  | 7                                   |
| `train_size` / `val_size`    | hotpotqa `build_spec`  | 50 / 100                            |
| `pydantic_agent_model`       | hotpotqa `build_spec`  | `"openai:gpt-4.1-mini"`             |
| `task_temperature`           | hotpotqa `build_spec`  | 0.0                                 |
| `max_concurrency`            | hotpotqa `build_spec`  | 4                                   |

GEPA-knob keys come from `OptimizationRunConfig` in
`rilixai/sandbox/runtime.py`; cookbook-knob keys are documented in
`_DEFAULT_SANDBOX_CONFIG` at the top of
`hotpotqa/optimization/spec.py`. Both kinds of keys ride together in
the same trigger config — rilixai's runtime plucks out the GEPA ones,
everything else lands in `ctx.config` for the spec factory to read
(`extra="allow"` on `OptimizationRunConfig` preserves them).

### Phase 3 — poll status and fetch results

```python
import time

while True:
    run = client.get_optimization_run(run_id)
    status = run["status"]
    print(f"  {status}", flush=True)
    if status in {"COMPLETED", "FAILED", "CANCELLED"}:
        break
    time.sleep(15)

# When COMPLETED, the best candidate + per-split eval reports are
# attached to the run record. The serialized PromptCandidate is the
# same shape `python -m hotpotqa.cli evaluate --candidate-json` reads,
# so you can pull it down and replay locally:
best = run["best_candidate"]
print("policy_prompt:\n", best["components"]["policy_prompt"][:500])
```

A 2000-metric-call HotpotQA run on `gpt-4.1-mini` typically takes
20–40 minutes wall-clock inside the sandbox (parallelism = 4 cases at
a time by default). Bump `max_concurrency` in the trigger config and
the matching value in rilixai's project-level rate limit to go faster;
bump `max_metric_calls` for tighter optimization at higher cost.

### Promoting / rolling back versions

For routine deploys, `--build` does push + promote in one shot — no
manual version management needed (each push registers
`v<short_sha>`, which becomes the new `@production`).

To roll back to a previously-pushed version without rebuilding:

```bash
uv run rilixai spec promote hotpotqa-agent --version v<older-sha>
```

To clear the promotion entirely (`@production` falls back to "latest
READY"):

```bash
uv run rilixai spec demote hotpotqa-agent
```

Trigger calls always reference `hotpotqa-agent@production`; rilixai
resolves to the currently promoted (or fallback) version server-side.
Override `--spec hotpotqa-agent@v<sha>` for one-off smoke / regression
runs without disturbing production.

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
