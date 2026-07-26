# HotpotQA

A self-contained **multi-hop question-answering agent** and a local evaluation
harness for it. The agent searches Wikipedia, chains evidence across several
paragraphs, and answers questions from
[HotpotQA](https://hotpotqa.github.io/) — then gets scored on exact match,
answer F1, and whether it actually retrieved the gold supporting paragraphs.

## The agent

A plain [PydanticAI](https://ai.pydantic.dev/) tool-using agent
(`src/hotpotqa/agent/agent.py`). One question in, one short answer out.

- **Task** — a HotpotQA question whose answer requires *hopping*: find one
  paragraph, learn a bridging entity from it, search again, then answer.
- **Tools** — two of them:
  - `retrieve_k(query)` — deterministic BM25 retrieval returning the top-`k`
    Wikipedia paragraphs not yet seen this case.
  - `summarize(question, passages, context=None)` — an LLM condensation of the
    retrieved passages. It's a direct `AsyncOpenAI.chat.completions` call (not a
    hidden sub-agent), so the prompt that lands in the API call is visible right
    at the call site. The agent decides whether to feed a previous summary back
    in as `context`.
- **Loop** — PydanticAI drives tool dispatch; the agent terminates by emitting
  the structured `HotpotQAOutput.answer`, so there is no `finish` tool.
  `max_iters` caps the number of model requests.
- **Prompts** — two, both overridable at construction (`agent/prompts.py`):
  `policy_prompt` (the agent's `system_prompt` / tool-use policy) and
  `summarize_prompt` (the summarize tool's system message).
- **Retrieval corpus** — pluggable via `--retrieval`:
  - `fullwiki` (default) — bm25s over the 2017 Wikipedia abstracts dump: the
    open-domain setting, where retrieval is a real part of the problem.
  - `distractor` — the HF `hotpot_qa[distractor]` per-case 10-paragraph
    context: cheap, offline-friendly, and the mode the tests use.

Code map:

Standard src layout — the importable package lives under `src/hotpotqa/`:

| Path | What it holds |
|---|---|
| `src/hotpotqa/agent/agent.py` | the PydanticAI agent, its two tools, and the trajectory it records |
| `src/hotpotqa/agent/prompts.py` | the two default prompts |
| `src/hotpotqa/agent/retrieval/` | the `retrieve_k` implementations (case-local BM25, fullwiki bm25s) |
| `src/hotpotqa/data/dataset.py` | HotpotQA loading + the typed `HotpotQARecord` |
| `src/hotpotqa/data/eval.py` | the canonical HotpotQA answer normalizer / EM / F1 |
| `src/hotpotqa/evaluation/scoring.py` | per-case field scoring + the weighted objective |
| `src/hotpotqa/evaluation/local_eval.py` | the bounded-concurrency batch evaluator |
| `src/hotpotqa/evaluation/report.py` | the JSON artifacts |
| `src/hotpotqa/config.py` | retrieval / model / loop-budget knobs |
| `src/hotpotqa/cli.py` | run the agent, or run + score it (`hotpotqa` console command) |

## How answers are scored

Three fields per case (`evaluation/scoring.py`):

- `answer` — exact match (0/1) after HotpotQA's canonical normalization
  (lowercase, strip articles and punctuation).
- `answer_f1` — token-level F1 against the gold answer, HotpotQA's denser
  answer metric.
- `supporting_titles_recall` — the fraction of the gold supporting paragraph
  titles the agent retrieved at any hop. This is the multi-hop signal: it says
  whether the evidence was ever found, independent of the final wording.

The objective is **pure exact match** by default
(`HOTPOTQA_FIELD_WEIGHTS`), matching how HotpotQA numbers are usually reported;
the other two fields are computed and reported as diagnostics. Pass different
`field_weights` to blend them.

The batch evaluator (`evaluation/local_eval.py`) runs cases concurrently,
bounded by `--max-concurrency`. One case failing never aborts the batch: it is
recorded with its error and counts as `0` (a real failure must deflate, never
inflate, the metrics). A case with no supervision at all (no gold answer and no
gold supporting titles) is **unscoreable** and dropped from the averages
instead.

## The data

Records come from the HuggingFace
[`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
dataset and are normalized into a plain typed `HotpotQARecord`.

`load_hotpotqa_paper_split` reproduces the standard benchmark slicing so
numbers stay comparable with published HotpotQA results: take the HotpotQA
*train* split (90k questions), slice `[0, 40%)` → test, `[40%, 80%)` →
validation, `[80%, 100%)` → train, then subsample each slice with
`random.Random(1)`. The defaults are the usual 300 test / 300 validation
cases.

## Install

Standalone [uv](https://docs.astral.sh/uv/) project; run everything from this
directory:

```bash
cd hotpotqa
uv sync --group dev
```

Provider key (needed for any run that calls a model):

```bash
export OPENAI_API_KEY=sk-...    # agent + summarize tool (gpt-4.1-mini default)
```

No dataset token is required — HotpotQA is public.

## Run

```bash
# Run the agent and dump its answers (no scoring):
uv run hotpotqa run \
    --split test --test-size 20 \
    --retrieval distractor \
    --output-dir hotpotqa_run

# Run the agent AND score it:
uv run hotpotqa evaluate \
    --split test --test-size 20 \
    --output-dir hotpotqa_run
```

Artifacts land in `--output-dir`:

| File | Contents |
|---|---|
| `run_outputs.json` | (`run`) per case: the answer, gold answer, retrieved titles, tool-call count — or an `error` |
| `eval_summary.json` | (`evaluate`) the objective, per-field means, and scored / errored / unscoreable counts |
| `eval_outputs.json` | (`evaluate`) the per-case scored results |

With no size flags, both commands use the 300-case fullwiki test slice. See
`--help` for all flags (`--split`, `--retrieval`, `--retrieve-k`,
`--max-iters`, `--max-concurrency`, `--task-model`, `--no-network`, …).

## Tests

```bash
uv run python -m pytest -q
```

Hermetic — a scripted PydanticAI `FunctionModel` plus a stubbed summarize call,
no network.

## Notes

- The 2017 Wikipedia abstracts dump (~5GB) downloads lazily on first `fullwiki`
  use and caches under `$XDG_CACHE_HOME/hotpotqa/fullwiki/` (override with
  `HOTPOTQA_FULLWIKI_CACHE_DIR`). Use `--retrieval distractor` to avoid it.
- Scoring supporting facts at *title* granularity (not sentence) matches what
  the agent retrieves: paragraphs. Sentence-level supporting-fact F1 can be
  layered on without changing the objective.
- `--no-network` makes the loaders refuse to download anything, so a
  misconfigured dry run can't quietly spend tokens.
