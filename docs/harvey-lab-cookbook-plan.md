# Plan: a Harvey LAB cookbook recipe

This document proposes a new cookbook recipe — **`harvey_lab`** — that turns
Harvey's open-source **Legal Agent Benchmark (LAB)** into a GEPA-optimizable
recipe, following the same shape as the existing `apex_agents` recipe. It also
records the **rilixai SDK migration** the cookbook needs before any new recipe
(or the existing two) can build against the current `@rilixai/rilixai` `main`.

> Scope: this is a design/plan document only. No recipe code is added here.

---

## 1. Background

### 1.1 What Harvey LAB is

Source: <https://github.com/harveyai/harvey-labs> (MIT-licensed).

LAB is a filesystem-first legal-work benchmark with two parts:

- **A task set** — `tasks/<practice-area>/<task-or-workflow>/[<scenario>/]`,
  each a directory with a `task.json` + a `documents/` folder of realistic
  legal source files (`.docx`, `.xlsx`, `.pptx`, `.pdf`, `.eml`).
- **An execution harness** — an agent loop (`harness/`) with six
  closed-workspace tools (`bash`, `read`, `write`, `edit`, `glob`, `grep`), a
  Podman sandbox, provider adapters, and an LLM-judge evaluator
  (`evaluation/`).

Scale (as cloned): **~1,749 tasks across 25 practice areas**; the biggest are
`contracts` (498), `corporate-ma` (161), `intellectual-property` (147). Tasks
carry **~60 rubric criteria on average** (min ~23, max ~194).

`task.json` fields:

| Field | Purpose |
|---|---|
| `title` | Human-readable task title |
| `instructions` | The directional prompt sent to the agent |
| `work_type` | `analyze` / `draft` / `review` / `research` (absent on the `contracts` subtree) |
| `deliverables` | Map of expected output filenames (absent on some tasks) |
| `criteria` | Inline list of `{id, title, deliverables, match_criteria}` pass/fail criteria |
| `tags` | Discovery/analysis metadata |

### 1.2 Scoring model

LAB uses **all-pass rubric scoring**: `score = 1.0 iff every criterion passed
else 0.0`. Each criterion is judged independently by an LLM judge, given the
task title, the criterion's `match_criteria`, and **only the deliverable files
that criterion lists** (falling back to all output when a criterion has no
`deliverables`). Criteria are graded `pass`/`fail` — there is no golden-answer
file; the `match_criteria` prose *is* the standard.

### 1.3 How this differs from `apex_agents` (important)

The existing `apex_agents` recipe and LAB look similar (toolbelt agent over a
document "world", LLM rubric judge) but differ in ways that drive the design:

| Dimension | `apex_agents` (Mercor APEX) | Harvey LAB |
|---|---|---|
| Dataset transport | Gated HF dataset `mercor/apex-agents` (zips + JSON) | Files committed in a **git repo** (no HF) |
| Agent output | A single `final_answer` **text string** | One or more **deliverable files** (`.xlsx`, `.docx`, …) on disk |
| Score | Fraction of rubric criteria met (continuous `rubric_pass_rate` in `[0,1]`) | **All-pass** binary per task (mean over tasks = all-pass rate) |
| Judge input | Task prompt + answer text + one criterion | Task title + **scoped deliverable file contents** + one criterion |
| Grouping axis | `world_id` (10 IB worlds) | `practice-area` (25) or task/workflow |
| Criteria/task | ~2–a few | **~60** (judge-call cost is the dominant knob) |

The **file-deliverable** output and the **all-pass** metric are the two biggest
departures from every existing recipe and are called out again in §4.

---

## 2. Anatomy of `apex_agents` (the template to mirror)

Each recipe is a uv workspace member with this layout. `harvey_lab` should
mirror it 1:1 so the cookbook stays uniform:

```
apex_agents/
  pyproject.toml            # member deps + rilixai pin + hatchling wheel remap
  README.md                 # reproduce commands, env vars, run-where table
  config.py                 # frozen dataclass of runtime knobs (models, caps, timeouts)
  cli.py                    # local laptop loop: optimize / evaluate / kfold subcommands
  sandbox.py                # push + promote + trigger one run on rilixai's Modal sandbox
  data/
    dataset.py              # load raw records -> normalize -> rilixai Case objects
    world_splits.py         # deterministic world-level train/val/kfold splitters
  agent/
    agent.py                # the faithful ReAct toolbelt agent (per-case build, lock-guarded)
    prompts.py              # seed prompt components + seed_candidate factory
    types.py                # framework-neutral dataclasses (AgentToolCall, AgentOutput)
    world/world.py          # on-disk world file surface + HF-backed world factory
  optimization/
    spec.py                 # build_apex_agents_spec(...) + @spec build_spec(ctx) sandbox factory
    runtime.py              # async ExtractionRuntime: run agent -> judge -> run_metrics
    metrics.py              # FieldConfig + MetricsCalculator + LLM rubric judge + verdict parser
    feedback.py             # per-component textual feedback strings for the reflection LM
  tests/                    # hermetic: FakeWorld + scripted model + stub judge, zero network
```

Key seams worth reusing:

- **Optimizable components.** `apex_agents` optimizes three prompt components:
  `system_prompt`, `task_template` (must preserve a `{{task}}` var),
  `resum_summary_prompt` (must preserve `{conversation}`). GEPA rewrites these;
  `feedback.py` gives the reflection LM per-component guidance and preservation
  guards.
- **World-level splits.** Splitting by *world* (not task) keeps train/val
  disjoint so GEPA selects for cross-world transfer. `world_splits.py` provides
  `fixed_val_split`, `stratified_case_cap` (round-robin so the world set stays
  wide at every train size), `world_held_out_val_split`, and `world_level_folds`.
- **Two entry points.** `cli.py` (local, `OPENAI_API_KEY` only) and
  `sandbox.py` (`rilixai push` → `spec promote` → `create_optimization_run`
  against `<name>@production`). CI (`.github/workflows/push-spec.yml`) runs
  `sandbox.py --build --no-trigger` per changed member.
- **Hermetic tests.** Inject a `FakeWorld` factory + scripted model + stub judge
  so the suite runs with zero network / zero spend.

---

## 3. Proposed `harvey_lab` recipe design

Add a `harvey_lab/` workspace member mirroring `apex_agents`, wired to the LAB
task set. Filenames below are the target; specifics may shift during build.

```
harvey_lab/
  pyproject.toml
  README.md
  config.py
  cli.py
  sandbox.py
  data/
    dataset.py              # load LAB task.json + documents -> normalized record -> Case
    task_splits.py          # deterministic practice-area-level splitters (reuse apex logic)
  agent/
    agent.py                # legal-work agent producing FILE deliverables in a workspace
    prompts.py              # seed prompt components + seed_targets factory
    types.py
    workspace/workspace.py  # per-task sandboxed workspace: documents/ (ro) + output/ (rw)
  optimization/
    spec.py
    runtime.py              # run agent -> collect deliverables -> all-pass judge -> metrics
    scoring.py              # per-criterion LLM judge + all-pass aggregation + verdict parse
    feedback.py             # per-component feedback for the reflection LM
  tests/                    # hermetic: FakeWorkspace + scripted model + stub judge
```

### 3.1 Dataset sourcing (`data/dataset.py`)

LAB ships as files in a git repo, not an HF dataset, so the loader differs from
`apex_agents`:

- **Option A (recommended): vendor a curated subset.** Copy a small,
  representative slice (e.g. 2–3 practice areas, N tasks each, with their
  `documents/`) under `harvey_lab/data/tasks/` and load from disk. Pros:
  hermetic, no gated access, deterministic CI, cheap. Cons: repo size; must
  track upstream refreshes. Keep a `MANIFEST`/`SOURCE.md` recording the upstream
  commit SHA the subset was cut from.
- **Option B: git submodule / pinned clone.** Add harvey-labs as a submodule (or
  a fetch step pinned to a commit) and read tasks from it. Pros: full task set,
  easy refresh. Cons: heavier checkout (~16k files), submodule friction, larger
  Modal image.
- **Option C: mirror to an HF dataset** (like `mercor/apex-agents`) and reuse
  the HF-download pattern. Pros: matches existing recipe transport; keeps the
  cookbook repo lean. Cons: extra hosting + a private/public licensing decision.

The loader normalizes each `task.json` into a `HarveyLabRecord` (`task_id` =
`<area>/<task>[/<scenario>]`, `title`, `work_type`, `instructions`,
`deliverables`, `criteria: tuple[Criterion]`, `document_paths`,
`practice_area`) and converts to a rilixai `Case` with `group_key =
practice_area` (so splitters stratify by area). Ground truth carries the rubric
criteria + deliverable map for the judge; the "prediction" is the set of output
files.

**Decision needed:** subset vs submodule vs HF (§7). Recommendation: start with
**Option A** for a hermetic first cut, leave a documented path to B/C.

### 3.2 The agent (`agent/agent.py`, `agent/workspace/`)

LAB's own harness already implements a capable file-producing agent (six tools,
Podman sandbox, provider adapters). Two build options:

- **Option 1 (recommended for a faithful, low-risk first cut): reimplement a
  faithful file-tool agent** inside the recipe, mirroring `apex_agents`'
  `ApexReActAgent` (per-case build, lock-guarded component snapshot, async
  `forward`). Give it read tools over `documents/` (docx/xlsx/pptx/pdf
  extraction, reusing LAB's extractors) and write/edit tools into an `output/`
  dir. Terminate when the model stops calling tools (LAB has no explicit
  `final_answer`), or on a step/turn cap. This keeps the recipe self-contained
  and hermetically testable with a scripted model + `FakeWorkspace`.
- **Option 2: wrap LAB's `harness.run_agent`** directly. Faster to reach parity
  with LAB numbers, but pulls in provider SDKs + Podman, complicates the Modal
  sandbox image and hermetic tests, and couples the recipe to LAB's harness
  internals. Prefer Option 1 unless exact LAB parity is a stated goal.

Either way, the **write/edit tools must land deliverables under a per-case
`output/` dir** that the runtime then scores. This is the core new capability
vs `apex_agents`.

### 3.3 Optimizable components (`agent/prompts.py`)

Propose optimizing:

- `system_prompt` — the legal-work agent's policy (read-before-writing, produce
  the named deliverables, cite sections, tool workflow, don't ask for docs that
  are in the workspace).
- `task_template` — the first user message wrapping `instructions` +
  `deliverables`; keep an `{{instructions}}` (and/or `{{deliverables}}`)
  substitution var with a preservation guard (mirrors the `{{task}}` guard).

Optionally a `planning_prompt` / summary component if a compaction step is
added. Start with two components; the seams generalize.

### 3.4 Runtime + scoring (`optimization/runtime.py`, `scoring.py`)

Per case: apply the target prompts → run the agent → collect the files written
to `output/` → for each rubric criterion, load its scoped deliverable file(s),
call the LLM judge for a `pass`/`fail` verdict → aggregate.

- **Per-task score = all-pass**: `1.0` iff every criterion passed, else `0.0`.
  (Consider *also* emitting a continuous `criterion_pass_rate` as a secondary,
  non-objective metric for smoother GEPA gradients — LAB reports pooled
  criterion pass rate in its dashboards; worth discussing in §7.)
- Reuse LAB's judge design: structured JSON verdict (`{verdict, reasoning}`),
  deliverable-scoped context, robust JSON parsing, retry-without-schema
  fallback. Port LAB's `_read_file_as_text` extractors (pandoc/pandas/
  markitdown/pdfplumber) for reading deliverables.
- Judge cost is the dominant knob given ~60 criteria/task — expose
  `--max-concurrency` for judge calls and a criteria cap for smoke runs.

### 3.5 Splits (`data/task_splits.py`)

Reuse `apex_agents/data/world_splits.py` almost verbatim, substituting
`practice_area` for `world_id` as the group key: hold out whole practice areas
for validation so GEPA selects for cross-area transfer; `stratified_case_cap`
round-robins across areas; `k`-fold partitions areas.

### 3.6 CLI + sandbox + CI

- `cli.py`: `optimize` / `evaluate` / `kfold`, `--no-network` guard, progress
  wrapper — copy `apex_agents/cli.py` structure.
- `sandbox.py`: `@spec(name="harvey-lab")` factory; `--build` → push + promote +
  trigger `harvey-lab@production`.
- Register the member in the **root `pyproject.toml`** (`[tool.uv.workspace]
  members`, `[tool.setuptools.packages.find] include`, `[tool.pytest]
  testpaths`, `[tool.mypy] files`/`exclude`) and add `harvey_lab` to the
  **`push-spec.yml`** path filter + matrix and a **`checks.yml`** build step.

---

## 4. Two behaviors with no precedent in the cookbook

Flagging these early because they need the most design attention:

1. **File deliverables as the prediction.** Every existing recipe returns a
   text/structured answer; LAB returns files. The runtime must give the agent a
   writable `output/` per case, capture what it wrote, and feed scoped files to
   the judge. The rilixai "prediction" becomes a manifest of output files (paths
   + extracted text), not a string.
2. **All-pass binary scoring.** The objective is `1.0`/`0.0` per task. With ~60
   criteria this is a *very* sparse signal for GEPA. Mitigations to weigh:
   emit `criterion_pass_rate` as an auxiliary field, start on tasks with fewer
   criteria, or make the objective the pass rate while reporting all-pass for
   parity. Needs a decision (§7).

---

## 5. rilixai SDK migration (required before any new recipe)

**The cookbook currently pins a very stale rilixai and imports a module
namespace that no longer exists on `main`.** Both existing recipes pin:

```
rilixai @ git+ssh://git@github.com/rilixai/rilixai.git@1baba33
```

`1baba33` ("PromptCandidate writeback… (#24)") is **~84 commits behind
`main`** (`main` HEAD `115028b`, rilixai **v0.2.0**). In between, the SDK's
public surface was **renamed and restructured** — the candidate/component
vocabulary became `OptimizationTargets`/`prompts`, and the spec contract was
slimmed to `Spec` + `data_loader`/`run_case`/`scorer` (see the table below).
The user's ask — *pin the pyproject to the latest `main` commit* — therefore
also requires a **code migration**, because the imports the recipes use are gone.

### 5.1 What broke

Everything the recipes import from `rilixai.prompt_optimization.*` moved to the
top-level `rilixai` surface / `rilixai.sdk`, with renamed types and a **new,
smaller spec contract** (`Spec` with `data_loader` + `run_case` + `scorer`,
instead of `PromptOptimizationSpec` with `extraction_runtime` +
`agent_resolver` + `field_extractor` + `evaluation_profile_resolver`).

| Old (pinned `1baba33`) | New (`main` / v0.2.0) |
|---|---|
| `from rilixai.prompt_optimization.models import Case` | `from rilixai import Case` |
| `PromptCandidate` | `OptimizationTargets` |
| `seed_candidate_from_components({...})` | `optimization_targets_from_prompts({...})` |
| `candidate.components` | `targets.to_dict()` (`{prompt_name: text}`) |
| `PromptOptimizationSpec(cases_by_split=, seed_candidate=, extraction_runtime=, agent_resolver=, field_extractor=, evaluation_profile_resolver=, task_type=, …)` | `Spec(name=, seed_targets=, data_loader=, run_case=, scorer=, evidence_provider=?, result_finalizer=?)` |
| `from rilixai.prompt_optimization.protocols import EvaluationProfile, FieldConfig, MetricsResult, ErrorOutput` | `from rilixai import FieldConfig, ErrorOutput` (+ `objective_score`); `EvaluationProfile`/`MetricsResult` replaced by the `CaseScorer` protocol + `CaseScore` |
| MetricsCalculator (`calculate_metrics`, field configs, comparators) | `CaseScorer.score_case(*, case, result) -> CaseScore` (per-case; rilixai aggregates field accuracy) |
| async `ExtractionRuntime` (`runtime(**kwargs)` reading `input`/`candidate`) | `async def run_case(*, case, targets, runtime) -> CaseResult` |
| `from rilixai.prompt_optimization.spec import OptimizationContext, build_adapter_from_spec, run_optimization_from_spec, validate_spec, PromptOptimizationRunConfig` | `from rilixai import OptimizationContext, validate_spec`; `build_adapter_from_spec` / `run_optimization_from_spec` / `PromptOptimizationRunConfig` **removed** (local optimize loop must be re-wired to the new runner API) |
| `from rilixai.prompt_optimization.evaluation import evaluate_candidate_on_cases, field_accuracy_rows, serialize_eval_outputs` | **removed** from the public surface (re-map to new evaluation helpers or drop the local `evaluate` path) |
| `from rilixai.prompt_optimization.optimization import extract_best_candidate, summarize_gepa_result_metadata` | **removed** from the public surface |
| Prediction/scoring dataclasses (`ApexAgentsRunResult`) | `CaseResult(output=, run_metrics=)` / `CaseScore(field_scores=, objective=, key=)` |
| Dataset = in-code `cases_by_split` dict | JSONL datasets (`STANDARD_JSONL_CASE_SCHEMA`, `train/val/test.jsonl`, `CaseDataLoader.parse_row`/`iter_cases`, `load_cases_by_split`, `materialize_dataset`) |

The canonical new spec-authoring shape is in the rilixai repo at
`packages/rilixai/src/rilixai/cli/init_templates.py` (the `rilixai init`
template) — it shows `@spec`, `Spec(...)`, a `CaseDataLoader`, an async
`run_case`, and a `CaseScorer` end-to-end. Use it as the migration reference.

### 5.2 Migration work items

1. **Bump the pin** in `hotpotqa/pyproject.toml` and `apex_agents/pyproject.toml`
   from `@1baba33` to the current `main` SHA (`115028b` at time of writing),
   then `uv lock` and commit the refreshed `uv.lock` (CI runs `--locked`).
2. **Rewrite the spec factories** (`optimization/spec.py`) to build a `Spec`
   with `data_loader` / `run_case` / `scorer` instead of `PromptOptimizationSpec`.
3. **Fold runtime + metrics into `run_case` + a `CaseScorer`.** The old
   `runtime.py` (`ExtractionRuntime`) becomes an async `run_case(*, case,
   targets, runtime)` returning `CaseResult`; `metrics.py`'s MetricsCalculator
   becomes a `CaseScorer.score_case` returning `CaseScore`.
4. **Rename the candidate/prompt types** throughout (`PromptCandidate` →
   `OptimizationTargets`, `seed_candidate_from_components` →
   `optimization_targets_from_prompts`, `.components` → `.to_dict()`).
5. **Re-wire the local CLI** `optimize`/`evaluate` paths — the old
   `run_optimization_from_spec` / `build_adapter_from_spec` /
   `evaluate_candidate_on_cases` / `PromptOptimizationRunConfig` /
   `extract_best_candidate` helpers are gone; map to the current runner + eval
   API (confirm exact names against `main`).
6. **Move datasets to JSONL** if the new runner requires it (the SDK now centers
   on `STANDARD_JSONL_CASE_SCHEMA` + `CaseDataLoader`). Decide whether the
   in-code `cases_by_split` path is still supported for local runs or whether
   recipes should emit `train/val/test.jsonl`.
7. **Update `reflection_evidence_mode` / feedback plumbing** to the new
   `EvidenceProvider` hook (`evidence_for(*, trajectory, diff)`) if the
   `curated_plus_trace` mode no longer exists as-is.
8. **Update tests** to the new imports/contracts, and both READMEs' "rilixai
   dependency" notes.

### 5.3 Sequencing recommendation

Do the migration as a **separate PR before** the `harvey_lab` recipe:

- **PR 1 — rilixai migration.** Bump the pin, migrate `hotpotqa` + `apex_agents`
  to the v0.2.0 `Spec`/`run_case`/`CaseScorer` contract, refresh `uv.lock`, get
  CI green. This is a self-contained, reviewable unit and de-risks the recipe.
- **PR 2 — `harvey_lab` recipe.** Build the new recipe **directly against the
  new SDK** (never against the old contract), reusing patterns from the
  already-migrated `apex_agents`.

Building `harvey_lab` first against the old contract would mean writing code
that's immediately obsolete, so the migration must lead.

---

## 6. Milestones

1. **M0 — rilixai migration (PR 1).** Pin bump + `hotpotqa`/`apex_agents`
   migrated to v0.2.0, `uv.lock` refreshed, CI green.
2. **M1 — dataset + splits.** `harvey_lab/data/` loads a vendored subset into
   `Case`s; hermetic tests for normalization + practice-area splits.
3. **M2 — agent + workspace.** File-producing agent over a sandboxed
   `documents/`(ro)+`output/`(rw) workspace; hermetic scripted-model test.
4. **M3 — scoring.** All-pass judge over scoped deliverables; verdict parsing +
   aggregation; stub-judge tests.
5. **M4 — spec + local CLI.** `Spec` factory + `run_case` + `CaseScorer`;
   `cli.py optimize/evaluate` runs end-to-end locally on a tiny slice.
6. **M5 — sandbox + CI.** `sandbox.py` push/promote/trigger; workspace member
   registered; `push-spec.yml` + `checks.yml` updated; README with reproduce
   commands.
7. **M6 — calibration.** Small real run; document a baseline all-pass rate and a
   smoke budget in the README (mirroring `apex_agents`' demo-bounded notes).

---

## 7. Open questions / decisions

1. **Dataset transport** — vendor a subset (A, recommended), git submodule (B),
   or mirror to HF (C)? Affects repo size, CI hermeticity, and licensing.
2. **Objective signal** — all-pass binary is very sparse with ~60 criteria/task.
   Optimize the continuous `criterion_pass_rate` while *reporting* all-pass for
   parity? Start on low-criteria tasks?
3. **Agent** — faithful reimplementation (1, recommended) vs wrapping LAB's
   `harness.run_agent` (2)? Trade self-containment/testability against exact LAB
   parity.
4. **Which practice areas / how many tasks** for the first cut? Suggest 2–3
   areas (e.g. `corporate-ma`, `contracts`) × a small N for a hermetic subset.
5. **Deliverable generation deps** — the agent must *write* `.xlsx`/`.docx`.
   Pull in `python-docx`/`openpyxl` (lazy-imported like `apex_agents`), or
   restrict the first cut to text/markdown deliverables?
6. **Judge model + cost** — reuse `gemini-2.5-flash` (APEX default) or
   `claude-sonnet` (LAB default)? Set a per-run criteria cap + concurrency for
   smoke runs.
7. **Exact new-runner API names** — confirm the current local optimize/evaluate
   entry points on `main` (the old `run_optimization_from_spec` /
   `evaluate_candidate_on_cases` helpers are gone) before writing `cli.py`.

---

## Appendix: quick references

- Existing recipe to mirror: `apex_agents/` (this repo).
- Upstream LAB: <https://github.com/harveyai/harvey-labs> — see its
  `docs/architecture.md`, `docs/eval-strategies.md`, `harness/agent_loop.py`,
  `harness/tools.py`, `evaluation/scoring.py`, `evaluation/judge.py`.
- New rilixai spec contract reference: rilixai
  `packages/rilixai/src/rilixai/cli/init_templates.py`,
  `packages/rilixai/src/rilixai/__init__.py`,
  `packages/rilixai/src/rilixai/sdk/` (`domain_contracts.py`,
  `spec_contract.py`, `models.py`).
- Current cookbook pin: `git+ssh://…/rilixai.git@1baba33`; target: `main` HEAD
  (`115028b`, v0.2.0).
