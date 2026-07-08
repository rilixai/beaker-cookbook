# Harvey LAB

Harvey's **Legal Agent Benchmark (LAB)** as a file-producing legal agent,
optimized by rilixai's GEPA loop.

- **Agent** — a [Stirrup](https://github.com/ArtificialAnalysis/Stirrup)
  tool-use loop (the harness Artificial Analysis'
  [Harvey LAB-AA leaderboard](https://artificialanalysis.ai/evaluations/harvey-lab-aa)
  runs on) over a per-task workspace: read-only `documents/` (the source
  record) and a writable `output/` for deliverables. Tools: `list_files`,
  `read_document` (.docx/.xlsx/.pdf/.eml/text), `grep_documents`,
  `write_deliverable`, `edit_deliverable`, `finish`.
- **Optimized prompts** — `system_prompt`, `task_template`.
- **Score** — **all-pass**: a per-criterion LLM judge grades each rubric
  criterion PASS/FAIL against its `match_criteria` (seeing only the
  deliverables that criterion names — deliverable-scoped), and a task scores
  `1.0` iff **every** criterion passes, else `0.0`. The dense
  `criterion_pass_rate` (fraction of criteria passed) is reported alongside
  and is the **optimizer objective** (see [Scoring](#scoring)).

Cases come from the public GitHub repo
[`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs), pinned to a
commit in `config.py`. The task documents are fetched at run time from that
commit (only the document filenames are stored in the dataset), so a spec
version stays reproducible as the upstream benchmark grows.

## Install

```bash
uv sync --all-packages --group dev
```

Env vars (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # agent (gpt-4.1-mini default)
export GOOGLE_API_KEY=...       # judge (gemini-2.5-flash default)
```

No dataset token is required — the LAB repo is public.

## Run locally

This recipe depends on the lightweight `rilixai` SDK only. The local CLI
covers the two SDK-only paths — `validate` (offline structural check) and
`evaluate` (score one candidate via the SDK `run_case` + scorer loop). The
full GEPA optimize loop runs server-side via `rilixai run` (see the Modal
section below).

```bash
# Validate the spec structure offline (no network, no documents, no LLM)
uv run python -m harvey_lab.cli validate

# Evaluate one candidate against a local checkout of the benchmark
git clone https://github.com/harveyai/harvey-labs
uv run python -m harvey_lab.cli evaluate \
    --tasks-root harvey-labs/tasks \
    --practice-areas contracts,corporate-ma \
    --test-size 10 \
    --candidate-json path/to/candidate.json    # omit to score the seed
```

`--val-areas` holds out whole practice areas when `--split validation` builds
the fixed val pool, so an evaluated candidate is scored for cross-practice-area
transfer rather than in-area fit. See `--help` for all flags.

## Scoring

`all_pass` (0/1) is the LAB-AA headline metric and is reported as-is. But on a
~60-criterion task it is an extremely sparse signal — a single missed
criterion zeroes the whole task — which gives GEPA almost no gradient. So the
optimizer objective is the **dense** `criterion_pass_rate` (fraction of
criteria passed); `all_pass` is tracked alongside as the metric of record. To
optimize all-pass directly instead, set
`field_weights={"all_pass": 1.0}` when building the spec.

## Run on Modal (rilixai sandbox)

`optimization/spec.py` registers a `@spec(name="harvey-lab")` factory that
rilixai's sandbox runs. `sandbox.py` builds the image, promotes it to
`harvey-lab@production`, and triggers a run in one shot.

**A dataset upload is required.** The spec sources its cases from an uploaded
JSONL dataset via `HarveyLabDataLoader` (see
`HarveyLabDataLoader.dataset_schema` in `harvey_lab/data/dataset.py`); a run
triggered with no dataset reference is rejected at startup. Export the split
from a local checkout, then upload:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/
export RILIXAI_AGENT_KEY=harvey-lab   # agent the trigger targets (or pass --agent)

git clone https://github.com/harveyai/harvey-labs
# --test-areas carves a disjoint test.jsonl the optimizer scores the winner on
# post-optimization (unbiased held-out); omit it for train/val only.
uv run python scripts/export_harvey_lab_dataset.py --tasks-root harvey-labs/tasks --val-areas 3 --test-areas 3
uv run rilixai dataset upload --name harvey-lab-dataset \
  --split train=883 --split val=123 --split test=126 scripts/_datasets/harvey_lab

uv run harvey_lab/sandbox.py --build   # build + promote + trigger
uv run harvey_lab/sandbox.py           # trigger only (current @production)
```

The trigger defaults to `--dataset harvey-lab-dataset@production` and
`--spec harvey-lab@production`; override either to pin a specific revision.
The `--max-metric-calls` budget is the primary cost knob (default 50, a smoke
budget). Provider keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`) are bound as
project-level secrets on rilixai's side and injected into each sandbox; the
task documents are fetched from the public LAB repo, so no dataset token is
needed at run time. Roll back with
`uv run rilixai spec promote harvey-lab v<older-sha>`.

## Tests

```bash
uv run python -m pytest harvey_lab/tests -q
```

Hermetic — a fixture task tree + a scripted Stirrup client + a stub judge, no
network.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- `max_turns=40` is demo-bounded; raise for parity with a full LAB-AA harness.
- Practice-area-level splits keep train / held-out legal domains disjoint.
