# MMDM decision agent

Answer [MMDM](https://arxiv.org/abs/2611.14309) questions with a decision model.
MMDM has 5,448 questions in 16 categories, from agent state tracking and tool
selection to policy exceptions and evidence validation. Each question is a
state, a question and a set of options: the native input of a decision API. The
seed agent sends each question to `jev-1.14.0` in one native TypeSafe request.

Beaker optimizes `prompts/` and `src/mmdm/agent.py`. The data, the request
guard and every model call (`src/mmdm/decide.py`), scoring
(`src/mmdm/scoring.py`) and the upstream [`mmdm_eval`](https://github.com/mmdm-benchmark/mmdm)
toolkit stay fixed.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run mmdm prepare --output /tmp/mmdm-data
uv run mmdm prepare --output /tmp/mmdm-agent-data \
  --categories agent_state_tracking,tool_skill_selection,workflow_decision,intent_routing,policy_exception,evidence_validation,constraint_planning
uv run beaker agent setup "MMDM" --integration-id mmdm
uv run beaker dataset upload /tmp/mmdm-data/optimization --name mmdm --split train=3212 --split test=1071
uv run beaker dataset upload /tmp/mmdm-agent-data/optimization --name mmdm-agent --split train=1373 --split test=457

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id mmdm --dataset 'mmdm@<revision>'
uv run beaker run trigger --integration-id mmdm --dataset 'mmdm@<revision>'
```

Upload only `optimization/`: its `test.jsonl` is validation data. Keep
`heldout/test.jsonl` for final evaluation of a frozen candidate.

A run with a model choice sends the decision calls to that model through
Beaker's gateway; without one they go to native Jev. Smoke checks structure
without model calls.

## Data

`prepare` downloads `mmdm.jsonl` from [`shenjunhao/mmdm`](https://huggingface.co/datasets/shenjunhao/mmdm)
at commit `1551e88c3b5290d6c5d3fb4c01b0ec00c57cfe7e`, checks its SHA-256, and
works offline after the first fetch. It fails if `gold` and
`answer.probabilities` disagree on any row.

MMDM ships one test set, so the recipe makes its own splits: a deterministic
hash split (seed `mmdm-v1`) stratified by category and label type, with about
60% train, 20% validation and 20% held out. Each file is ordered by a seeded
hash, so any prefix is a fair sample. `manifest.json` records the source, seed
and counts per split and category.

| Dataset | Categories | Train | Validation | Held out |
|---|---|---:|---:|---:|
| `mmdm` | all 16 | 3,212 | 1,071 | 1,071 |
| `mmdm-agent` | the 7 listed above | 1,373 | 457 | 457 |

**94 questions are left out.** They have more than 255 options, the most one
decision request accepts, and the request guard requires every option in every
request. 78 are `constraint_planning` and 16 are `tool_skill_selection`; the
manifest lists their IDs. So results here cover 5,354 of MMDM's 5,448 questions
and are not directly comparable to the paper's.

Each row is `{"id", "input": {state, question, options, task, score_values},
"expected": {label_semantics, probabilities, correct_option_ids, task_category,
task_subcategory, source}}`. Only `input` reaches the agent.

## The agent

The seed puts `prompts/instructions.md` in front of the question's own
instructions, cuts states that exceed the request budget with upstream
head–tail truncation (32,768-token context, 1,024 service overhead, upstream's
pinned proxy tokenizer), and makes one decision call. Upstream
`mmdm_eval.common.wire` builds the request: `choice`, `noul` and `score`
questions go natively; `multichoice` goes as `choice`, and scoring takes the
top N.

A candidate may add text to the instructions or the state, pre-process the
state, make several decision calls, or add chat calls (`session.chat`, traced and
following the run's model). `decide.py` rejects any request that drops, adds,
renames or edits an option, changes the task, or removes the question text. The
agent never sees labels or category.

## Scoring

Each question is scored with upstream `mmdm_eval.metrics.score`:

| Label type | Objective |
|---|---|
| hard single (4,046) | 1 if the unique top option is correct; a tie for first is wrong |
| hard set (605) | 1 if every gold option ranks strictly above every other option |
| soft (703) | 1 − total variation distance |

Field scores, where they apply: `hard_accuracy`, `set_match`, `brier`,
`total_variation`, `score_nmae` (score tasks) and `top_probability`. Field
scores must lie in [0, 1], so `brier` is upstream's sum-of-squares Brier divided
by its maximum of 2; the `evaluate` summary reports soft Brier on upstream's
[0, 2] scale. A request
error, malformed answer or missing option fails the case with its error
recorded.

## Evaluate locally

Set `TYPESAFE_API_KEY` in your shell; this command makes paid API calls:

```bash
uv run mmdm evaluate --data /tmp/mmdm-data --split validation --limit 50 \
  --output /tmp/mmdm-validation.jsonl
```

Use `--split test` with no `--limit` for the held-out evaluation and
`--prompts <directory>` for a candidate's prompts. The command writes per-case
results, `<output>.summary.json`, and `<output>.upstream-predictions.jsonl`,
then scores those predictions with upstream `score_predictions` and fails if the
numbers differ from its own. Any failed case makes it exit non-zero. To score
with the upstream command too, pass the matching rows of `mmdm.jsonl` as
`--input`:

```bash
uv run mmdm-score --input <rows.jsonl> --predictions /tmp/mmdm-validation.upstream-predictions.jsonl --output <dir>
```

The summary reports hard accuracy (overall and per category), set match, soft
Brier, score nMAE, ECE over hard single questions (10 bins), coverage, p50/p95
latency, and cost per 1,000 questions at Jev's reported $0.042 per million input
tokens.

Run offline tests with `BEAKER_AGENT_KEY= uv run python -m pytest -q`.

## Seed baseline

Not yet measured: this needs a TypeSafe API key and paid calls on validation
and held out, for `mmdm` and `mmdm-agent`.

## References

- Shen et al. (2026), [*MMDM: Measuring Massive Multitask Decision Making*](https://arxiv.org/abs/2611.14309).
- Data: [`shenjunhao/mmdm`](https://huggingface.co/datasets/shenjunhao/mmdm). Toolkit: [`mmdm-benchmark/mmdm`](https://github.com/mmdm-benchmark/mmdm) at `eedc865ef7303740a92e1abe73e13698e9f88091`, installed unmodified.

See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).
