# HotpotQA

A self-contained multi-hop question-answering agent — and how rilixai's GEPA
loop optimizes its prompts. This folder is one integrated demo: a realistic
agent, its data, its grading, and its rilixai integration, with no shared
cookbook code.

## The agent

The agent answers **multi-hop** questions — questions that require chaining
facts from more than one Wikipedia article — by retrieving passages and
reasoning over them.

- **Input** — one question string per case (plus, for grading, the gold
  answer and supporting facts, which the agent never sees).
- **Tools** — a two-tool PydanticAI toolbelt:
  - `retrieve_k(query)` — deterministic BM25 paragraph retrieval, no LLM.
  - `summarize(question, passages, context=None)` — a raw `AsyncOpenAI` call
    (kept raw so the optimized prompt is visible right at the call site).
- **Loop** — a PydanticAI tool-using agent: it issues `retrieve_k` /
  `summarize` calls in a loop (up to `max_iters`) and terminates by returning
  a structured Pydantic `HotpotQAOutput` (`answer` + `supporting_facts`).
- **Model** — a PydanticAI model string (default `openai:gpt-4.1-mini`) for
  the agent loop; `summarize` uses the same model via LiteLLM/OpenAI.
- **Output** — the final `HotpotQAOutput.answer` string.
- **Prompts** — two of them: `policy_prompt` (the agent's `system_prompt`)
  and `summarize_prompt` (the summarize tool's system message). These are the
  strings rilixai optimizes.

The importable package lives under `src/hotpotqa/` (standalone src-layout
project). Code map: `src/hotpotqa/agent/agent.py` (the PydanticAI loop),
`src/hotpotqa/agent/retrieval/` (BM25 / fullwiki retrievers),
`src/hotpotqa/agent/prompts.py` (the seed prompts), `src/hotpotqa/config.py`
(model / retrieval knobs). Paths below are relative to `src/hotpotqa/` unless
noted.

### How the work is graded

Each case ships a gold answer. The scorer computes **exact-match** (0/1) and
token **F1** of the agent's answer against the gold, plus a
`supporting_facts` overlap signal (`optimization/metrics.py`). The recipe
optimizes the dense F1 signal and reports exact-match alongside.

### The data (and its splits)

Retrieval is pluggable and picks the data source:

- `distractor` — HuggingFace `hotpot_qa[distractor]`, 10 paragraphs/case.
  Fast and test-friendly; the default.
- `fullwiki` — paper parity: `bm25s` over the 2017 Wikipedia abstracts dump
  (~5GB, downloaded lazily on first use and cached under
  `$XDG_CACHE_HOME/rilixai/hotpotqa/fullwiki/`).

HotpotQA is a **public** benchmark. Slicing is bit-faithful to the GEPA
artifact: the HotpotQA *train* split (90k cases) is sliced `[0, 40%)` → test,
`[40%, 80%)` → val, `[80%, 100%)` → train, sampled with `random.Random(1)`;
the 300/300/150 picks match the paper. Treat the scores as a
**before/after-optimization delta on this fixed harness**.

## How rilixai plugs in

Everything rilixai-specific lives in `optimization/`. The integration is one
`Spec` binding four things (`optimization/spec.py`):

| Spec field | What it is here |
|---|---|
| `seed_targets` | the two seed prompts (`agent/prompts.py`) |
| `data_loader` | `HotpotQADataLoader` — maps one uploaded JSONL row (raw `hotpot_qa` record) → one `Case` (`data/dataset.py`) |
| `run_case` | async adapter that runs the agent on a case and returns a `CaseResult` (`optimization/runtime.py`) |
| `scorer` | `HotpotQAScorer` — exact-match / F1 against the gold answer (`optimization/metrics.py`) |

The optimizer drives `run_case(case=..., targets=..., runtime=...)` with each
candidate prompt bundle; GEPA keeps the prompts that score higher.

**Optional model selection.** `run_case` respects `runtime.model` when the
optimizer selects a model for a rollout (overriding `pydantic_agent_model`),
and otherwise uses the recipe's own default. Ordinary runs need no model from
rilixai.

**Recipe knobs travel under `extra`.** The strict launch config only accepts a
fixed set of top-level keys (`max_metric_calls`, …); this recipe's own knobs
(`retrieval_mode`, `retrieve_k`, `max_iters`, `pydantic_agent_model`,
`task_temperature`) travel under the launch config's `extra` block, which
`build_spec` reads via `ctx.config["extra"]` (see `_DEFAULT_SANDBOX_CONFIG` at
the top of `optimization/spec.py`).

**Local vs hosted.** This recipe depends only on the lightweight `rilixai`
SDK. The local CLI (`cli.py`) covers the two SDK-only paths — `validate` and
`evaluate` (`optimization/local_eval.py` scores one candidate the way the
hosted runner does). The full GEPA optimize loop runs server-side via
`rilixai run`; `sandbox.py` builds + promotes + triggers it.

## Install

This recipe is a standalone uv project — set it up from its own folder:

```bash
cd hotpotqa
uv sync --group dev
```

Env vars (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # agent + summarize (gpt-4.1-mini default)
```

## Run locally

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

**A dataset upload is required.** The spec sources its cases from an uploaded
JSONL dataset via `HotpotQADataLoader` (row schema: raw `hotpot_qa` records;
see `HOTPOTQA_DATASET_SCHEMA` in `data/dataset.py`). A run triggered with no
dataset reference is rejected at startup. Upload a split directory once, then
trigger:

```bash
export RILIXAI_API_KEY=sk-...
export RILIXAI_API_BASE_URL=https://<id>.execute-api.<region>.amazonaws.com/prod/
export RILIXAI_AGENT_KEY=hotpotqa-agent   # agent the trigger targets (or pass --agent)

# One-time (or when the data changes): upload the JSONL split as a dataset.
uv run rilixai dataset upload --name hotpotqa-agent-dataset path/to/jsonl-dir/

uv run sandbox.py --build   # build + promote + trigger
uv run sandbox.py           # trigger only (current @production)
```

The trigger defaults to `--dataset hotpotqa-agent-dataset@production` and
`--spec hotpotqa-agent@production`; override either to pin a specific revision.
`OPENAI_API_KEY` is bound as a project-level secret on rilixai's side, injected
into each sandbox. The train/val split is derived from the uploaded dataset
server-side, so there are no `train_size`/`val_size` knobs. Roll back with
`uv run rilixai spec promote hotpotqa-agent v<older-sha>`.

CI (`.github/workflows/push-spec.yml`) runs `sandbox.py --build --no-trigger`
on every merge to `main` that touches `hotpotqa/`: it ships the image and
flips `@production` without spending LLM tokens on a smoke run.

## Tests

```bash
uv run python -m pytest -q
```

Hermetic — scripted PydanticAI `FunctionModel`s, no network.

## Notes

- rilixai is pinned via git in `pyproject.toml` until it's on PyPI.
- The 2017 Wikipedia abstracts dump (~5GB) downloads lazily on first
  `fullwiki` use and caches under `$XDG_CACHE_HOME/rilixai/hotpotqa/fullwiki/`.
