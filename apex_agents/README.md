# APEX-Agents

The Mercor **APEX-Agents** benchmark (Law / Investment Banking) wrapped as a
native async ReAct toolbelt agent, with rilixai's GEPA loop optimizing its
three prompt components.

* **Tools**
  * **Meta** — `final_answer`, `request_more_steps`, `request_more_cost`,
    `resum` (conversation summarization with an optimizable prompt).
  * **Domain** — a typed toolbelt over the per-case world (zip-extracted
    files): `list_files`, `read_text_file`, `read_pdf`, `read_docx`,
    `read_xlsx`, `search_files`, …
* **Optimizable components**
  * `system_prompt` — high-level ReAct policy + tool-use guidance.
  * `task_template` — wraps the per-case prompt (with `{{task}}` slot).
  * `resum_summary_prompt` — instructs the `resum` tool on how to compress
    the conversation when the context budget tightens.
* **Terminator** — the agent calls `final_answer(answer=…)`.
* **Score** — `rubric_pass_rate` from an LLM judge (Gemini 2.5 Flash by
  default) graded against the per-task rubric.

Cases come from the private HF dataset `mercor/apex-agents`. Each case has:

* a domain (`law` / `investment_banking`),
* a `world_id` pointing at a zipped folder of files (statutes, deal rooms,
  filings, …),
* a task prompt + a rubric of LLM-judged criteria.

World extraction is lazy — `openpyxl` / `pypdf` / `python-docx` / `xlrd` are
only imported when a tool actually opens a corresponding file. Tests use a
`FakeWorld` shim so the suite stays hermetic.

## Install

From the cookbook repo root:

```bash
uv sync --all-packages --group dev
```

(The cookbook is a uv workspace; `--all-packages` installs every benchmark
member.)

Required env vars before running anything that calls a model:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...   # private dataset access
export OPENAI_API_KEY=sk-...           # task LM (gpt-4.1-mini default)
export GOOGLE_API_KEY=...              # judge LM (gemini-2.5-flash default; GEMINI_API_KEY also works)
```

(Or `ANTHROPIC_API_KEY` etc., depending on `--task-model` / `--judge-model`.)

## Run

### Optimize on a small Law slice

```bash
uv run python -m apex_agents.cli optimize \
    --domain law \
    --train-size 25 \
    --val-size 20 \
    --val-worlds 2 \
    --max-metric-calls 100 \
    --output-dir apex_agents_results/smoke
```

The `--val-worlds` flag holds out entire worlds for inner validation so GEPA
selects for cross-world transfer, not in-world fit (the original Law fold-0
sweep showed val→test collapse without this).

For a heavier run, bump `--max-metric-calls` and `--train-size`. Pass
`--domain investment_banking` for the IB slice.

### Evaluate a candidate

```bash
uv run python -m apex_agents.cli evaluate \
    --domain law \
    --candidate-json apex_agents_results/smoke/best_candidate.json \
    --output-dir apex_agents_results/smoke-eval
```

With no `--candidate-json`, evaluates the three seed prompts unchanged
(paper baseline).

## Run on Modal (rilixai sandbox)

The cookbook has two entry points:

| Script | Where it runs | When to use |
|---|---|---|
| `apex_agents/cli.py`     | Your laptop                  | Local debugging, evaluating a downloaded candidate, no rilixai account needed |
| `apex_agents/sandbox.py` | rilixai hosted Modal sandbox | Real optimization runs, sweeps, scheduled retraining, sharing candidates across teams |

(A follow-up PR will unify these into one subcommand-style CLI.)

`apex_agents/optimization/spec.py` registers a `@spec(name="apex-agents")`
factory that rilixai's sandbox dispatcher invokes once per run. No version
is pinned in the decorator — `sandbox.py --build` supplies `v<short_sha>` at
push time and **promotes the freshly-pushed row to
`apex-agents@production`** in the same invocation. Triggers reference
`apex-agents@production`, which rilixai resolves server-side to whatever's
currently promoted.

This mirrors the [rilix prod-latest pattern](https://github.com/rilixai/rilix/pull/1381)
— no manual version bumps in source for routine deploys, and rollback is
`uv run rilixai spec promote apex-agents --version <older>`.

### Prerequisites

```bash
export RILIXAI_API_KEY=sk-...                                            # rilixai control-plane credential
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/   # CDK stack ApiUrl output
```

The provider keys the agent + judge need (`OPENAI_API_KEY`,
`GOOGLE_API_KEY`, …) and `HUGGING_FACE_HUB_TOKEN` are bound at the *project*
level on rilixai's side, not on your machine — see rilixai's docs for
project-secret configuration. Once bound, every run spawned under that
project receives them as env vars inside the sandbox.

**Scope.** Every optimization run belongs to a `scope_key` — a
customer-defined stable string (e.g. `"apex-agents"`) that groups runs +
releases. The scope row is auto-created on first reference, so just pick a
value and pass it to every `create_optimization_run` call.

### One-shot: build + promote + trigger via `sandbox.py`

The canonical CI-equivalent flow:

```bash
uv run apex_agents/sandbox.py --build
```

What this does:

1. **Push** the cookbook source as a fresh Modal image, registered as
   `apex-agents@v<short_sha>` (the short SHA comes from
   `git rev-parse --short HEAD` automatically).
2. **Promote** the just-pushed version: `apex-agents@production` now
   resolves to it.
3. **Trigger** a run referencing `apex-agents@production`.

For trigger-only flows (no rebuild needed — pick up whatever's currently
`@production`):

```bash
uv run apex_agents/sandbox.py
```

To pin a specific build for smoke / regression debugging without flipping
production:

```bash
# Push a fresh build but don't promote it:
uv run apex_agents/sandbox.py --build --no-promote

# Then trigger against that specific version:
uv run apex_agents/sandbox.py --spec apex-agents@v1a2b3c4
```

### Phase 1 — push the spec (image build)

`rilixai push` bundles the cookbook source tree, ships it to rilixai's
build worker, which installs the `apex_agents` workspace member's deps into
a Modal image and registers `apex-agents@v<short_sha>` against your
project. Re-run only when code or deps change.

```bash
cd ~/Projects/rilixai-cookbook
SPEC_VERSION="v$(git rev-parse --short HEAD)"
uv run rilixai push \
    --source-dir . \
    --name apex-agents \
    --version "$SPEC_VERSION" \
    --pip-install "litellm>=1.74" \
    --pip-install "huggingface-hub>=0.20" \
    --pip-install "openpyxl>=3.1" \
    --pip-install "pypdf>=4.0" \
    --pip-install "python-docx>=1.1" \
    --pip-install "xlrd>=2.0" \
    apex_agents/optimization/spec.py

# Then promote to production:
uv run rilixai spec promote apex-agents --version "$SPEC_VERSION"
```

The `--pip-install` entries are required because the build worker runs
`pip install /spec` against the **cookbook root** `pyproject.toml` (which
declares zero deps — it's just a uv workspace marker). The workspace
member's deps live in `apex_agents/pyproject.toml` and are invisible to the
build worker, so they have to be passed explicitly. `sandbox.py --build`
does this for you by reading the member pyproject at runtime.

(The spec-file path is positional, not a flag — see `rilixai push --help`.)

What happens server-side:

1. Source tree zipped + uploaded to a presigned S3 URL.
2. Build worker bakes a Modal image: base Python 3.12 + `pip install
   ./apex_agents` (pulls rilixai via the git+ssh pin + litellm +
   huggingface-hub + the lazy parsers + …).
3. `SpecVersion` row stamped with `status=READY` and `image_ref=<modal
   image id>`. The CLI polls until READY and exits.
4. `spec promote` flips `apex-agents@production` to point at the
   just-pushed row.

First-time builds take ~3–8 minutes. Subsequent pushes with cached layers
finish in under a minute. Older versions stay addressable forever
(`spec="apex-agents@v<older-sha>"`) so historical runs are reproducible;
rollback is `uv run rilixai spec promote apex-agents --version <older>`.

### Phase 2 — trigger a run

```python
from rilixai import RilixAIClient

client = RilixAIClient(
    base_url="https://api.rilix.ai",
    api_key="sk-...",     # your RILIXAI_API_KEY
)

response = client.create_optimization_run(
    task_type="apex_agent",
    spec="apex-agents@production",  # resolves to the currently promoted version
    scope_key="apex-agents",         # customer-defined stable key; auto-created on first use
    config={
        # ─── GEPA per-run knobs (consumed by rilixai's sandbox runtime) ─────
        "max_metric_calls": 100,              # ⚠ GEPA budget — primary cost knob
        "reflection_minibatch_size": 3,
        "reflection_model": "openai/gpt-4.1", # paper-parity reflection LM
        "seed": 0,

        # ─── APEX-Agents cookbook knobs (consumed by build_spec in spec.py) ─
        "domain": "law",                       # or "investment_banking"
        "train_size": 25,                      # stratified across train worlds
        "val_size": 20,                        # drawn from held-out worlds
        "val_worlds": 2,                       # whole worlds held out for inner val
        "task_model": "openai/gpt-4.1-mini-2025-04-14",
        "task_temperature": 0.0,
        "judge_model": "gemini/gemini-2.5-flash",
        "max_steps": 60,
        "cost_limit": 3.0,
    },
)
run_id = response["id"]
print(f"queued run: {run_id}")
```

**Where `max_metric_calls` is set** (the question every cost-conscious
caller asks first): inside the trigger's `config={…}` dict, as a top-level
key. rilixai's sandbox runtime parses it directly into the GEPA
`PromptOptimizationRunConfig` — **`build_spec` does not see or control
it**. The split between "GEPA knobs" and "APEX-Agents cookbook knobs" in
the config above mirrors that boundary:

| Key                          | Owned by                  | Default if omitted                  |
|------------------------------|---------------------------|-------------------------------------|
| `max_metric_calls`           | rilixai sandbox           | 50 (rilixai default — bump it!)     |
| `reflection_minibatch_size`  | rilixai sandbox           | 3                                   |
| `reflection_model`           | rilixai sandbox           | (reuses task model)                 |
| `seed`                       | rilixai sandbox           | 0                                   |
| `domain`                     | apex_agents `build_spec`  | `"law"`                             |
| `train_size`                 | apex_agents `build_spec`  | 25                                  |
| `val_size`                   | apex_agents `build_spec`  | 20                                  |
| `val_worlds`                 | apex_agents `build_spec`  | 2                                   |
| `task_model`                 | apex_agents `build_spec`  | `"openai/gpt-4.1-mini-2025-04-14"`  |
| `task_temperature`           | apex_agents `build_spec`  | 0.0                                 |
| `judge_model`                | apex_agents `build_spec`  | `"gemini/gemini-2.5-flash"`         |
| `max_steps`                  | apex_agents `build_spec`  | 60                                  |
| `cost_limit`                 | apex_agents `build_spec`  | 3.0                                 |
| `max_concurrency`            | apex_agents `build_spec`  | 4                                   |

GEPA-knob keys come from `OptimizationRunConfig` in
`rilixai/sandbox/runtime.py`; cookbook-knob keys are documented in
`_DEFAULT_SANDBOX_CONFIG` at the top of `apex_agents/optimization/spec.py`.
Both kinds of keys ride together in the same trigger config — rilixai's
runtime plucks out the GEPA ones, everything else lands in `ctx.config` for
the spec factory to read (`extra="allow"` on `OptimizationRunConfig`
preserves them).

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
# same shape `python -m apex_agents.cli evaluate --candidate-json` reads,
# so you can pull it down and replay locally:
best = run["best_candidate"]
print("system_prompt:\n", best["components"]["system_prompt"][:500])
```

A 100-metric-call APEX-Agents Law smoke on `gpt-4.1-mini` typically takes
~25–45 minutes wall-clock inside the sandbox (parallelism = 4 cases at a
time by default; the ReAct loop runs up to `max_steps=60` per case so
individual cases are slower than HotpotQA). Bump `max_concurrency` in the
trigger config and the matching value in rilixai's project-level rate
limit to go faster; bump `max_metric_calls` for tighter optimization at
higher cost.

### Promoting / rolling back versions

For routine deploys, `--build` does push + promote in one shot — no manual
version management needed (each push registers `v<short_sha>`, which
becomes the new `@production`).

To roll back to a previously-pushed version without rebuilding:

```bash
uv run rilixai spec promote apex-agents --version v<older-sha>
```

To clear the promotion entirely (`@production` falls back to "latest
READY"):

```bash
uv run rilixai spec demote apex-agents
```

Trigger calls always reference `apex-agents@production`; rilixai resolves
to the currently promoted (or fallback) version server-side. Override
`--spec apex-agents@v<sha>` for one-off smoke / regression runs without
disturbing production.

## Tests

```bash
uv run python -m pytest apex_agents/tests -q
```

Tests run hermetically against a `FakeWorld` shim + a stub rubric judge —
no network access required (no HF download, no LiteLLM calls). The
behavioral ReAct-loop suite lives in `tests/test_agent.py`; everything else
(dataset normalization, k-fold, metrics, spec wiring, per-component
feedback) is in `tests/test_units.py`.

## Notes

* The cookbook installs rilixai from a git+ssh pin in
  `apex_agents/pyproject.toml`. When rilixai is published to PyPI at a
  fresh version, the pin swaps to `rilixai==X.Y.Z`.
* The `mercor/apex-agents` dataset is private — CI auth flows through an
  ssh key (`RILIXAI_GITHUB_SSH_KEY` via `webfactory/ssh-agent`), matching
  the rilix repo's pattern. Local runs need
  `huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN` once.
* World extraction caches under `$XDG_CACHE_HOME/rilixai/apex_agents/` so
  the same world isn't re-unzipped across cases that share it.
* `max_steps=60` and `cost_limit=$3` are demo-bounded — the Archipelago
  paper uses 250 steps / much higher cost ceilings. Raise for production
  parity; lower for smoke loops.
