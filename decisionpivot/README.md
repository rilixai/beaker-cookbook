# DecisionPivot decision agent

Make the control-point decisions of an LLM system with a decision model, on the
core track of [DecisionPivot](https://arxiv.org/abs/2610.28522). The 13 core
families ask what an agent harness asks: can this request be answered, is this
action allowed, is this tool call valid, did the trace reach the goal, which
record breaks the rule. Each item is a state and one typed question, the native
input of a decision API. The seed agent sends each item to `jev-1.14.0` in one
native TypeSafe request.

Every item belongs to a **contrast group**: the same case two to four times,
with one planted detail that changes the answer. One Beaker case is one group,
and the objective is **pair correct**: 1 only if every member is right.

Beaker optimizes `prompts/` and `src/decisionpivot/agent.py`. The data, the
request guard and every model call (`src/decisionpivot/decide.py`), scoring
(`src/decisionpivot/scoring.py`) and the upstream generators and scorer stay
fixed.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run decisionpivot prepare --output /tmp/decisionpivot-data
uv run beaker agent setup "DecisionPivot" --integration-id decisionpivot
uv run beaker dataset upload /tmp/decisionpivot-data/optimization --name decisionpivot \
  --split train=2676 --split test=1992

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id decisionpivot --dataset 'decisionpivot@<revision>'
uv run beaker run trigger --integration-id decisionpivot --dataset 'decisionpivot@<revision>'
```

For a cheap trial run first, build a small dataset with the first groups of
each family (4 per family: 52 groups and about 128 items per split):

```bash
uv run decisionpivot prepare --output /tmp/decisionpivot-small --groups-per-family 4
uv run beaker dataset upload /tmp/decisionpivot-small/optimization --name decisionpivot-small \
  --split train=52 --split test=52
```

Upload only `optimization/`: its `test.jsonl` is validation data. Keep
`heldout/test.jsonl` for final evaluation of a frozen candidate.

A run with a model choice sends the decision calls to that model through
Beaker's gateway; without one they go to native Jev. Smoke checks structure
without model calls.

## Data

DecisionPivot's core track has 7,800 items: 13 families, 600 items each, 200 on
each of three levels. `prepare` builds them from the pinned upstream generators;
the Hugging Face download is not needed. It fetches
[`StartLuxLabs/DecisionPivot`](https://github.com/StartLuxLabs/DecisionPivot) at
commit `fd57688f033864e64c452561ffdfcba9739d1811` into `~/.cache/decisionpivot/`,
runs `generators/regenerate.py --check` there unmodified, and fails unless every
items file matches its pinned SHA-256. The `--check` run also runs upstream's
independent solvers, format check, content-blind audit and key-balance check.
Generating takes about a minute. The reference track (35 public test sets) is
not used.

All 13 families reproduce the release 1.1 hashes on CPython 3.12.12 (upstream
recorded them with 3.9.6).

| layer | family | question | options |
|---|---|---|---:|
| A | `answerable_facts` | can the facts answer the question | 2 |
| B | `perm_inherit` | may the user act on the folder | 2 |
| B | `policy_case` | approve, deny or escalate | 3 |
| C | `log_count_check` | is the claimed event count right | 2 |
| D | `sched_conflict` | can the booking be confirmed | 2 |
| D | `state_goal` | does the trace reach the goal | 2 |
| D | `toolcall_valid` | is the tool call valid | 2 |
| D | `triage_order` | which case is at the asked place | 5 |
| E | `clause_change` | does the edit change the answer | 3 |
| E | `dep_resolve` | do the pinned versions install | 2 |
| F | `ledger_reconcile` | which entry fails, or none | 6 |
| F | `record_rule` | which rule is broken, or none | 7–10 |
| G | `filter_match` | does the record pass every filter | 2 |

### Splits

One row is one contrast group, and a group never crosses files.

- **Seven families take a second seed** (`clause_change`, `dep_resolve`,
  `ledger_reconcile`, `perm_inherit`, `record_rule`, `sched_conflict`,
  `triage_order`): train is a fresh draw with seed 1, validation a fresh draw
  with seed 2, held out the published release 1.1 items. Fresh draws share no
  item or group IDs with the published items.
- **Six families cannot be redrawn** (`answerable_facts`, `filter_match`,
  `log_count_check`, `policy_case`, `state_goal`, `toolcall_valid`; their
  generators have a fixed seed). Their published groups are split by a seeded
  hash (seed `decisionpivot-v1`) within each family and level: about 60% train,
  20% validation, 20% held out. **These six families are optimized on the same
  pool of published items they are tested on** (different groups, same
  generator draw).

| File | Groups | Items |
|---|---:|---:|
| `optimization/train.jsonl` | 2,676 | 6,367 |
| `optimization/test.jsonl` (validation) | 1,992 | 4,916 |
| `heldout/test.jsonl` | 1,992 | 4,917 |

Each file is ordered by a seeded hash, so any prefix is a fair sample.
`manifest.json` records the upstream commit, seeds, Python version, every
items-file hash, and counts per split, family and level.

Row shape:

```json
{"id": "dp-<group id>",
 "input": {"family": "...", "items": [{"id": "...", "state": "..." | {...}, "questions": {"q": {...}}}]},
 "expected": {"gold": {"<item id>": "<option key>"}, "level": 1, "layer": "D"}}
```

Only `input` reaches the agent. Upstream's `meta` (including `spec` and
`deviation`, which reveal the answer) is dropped.

## The agent

The agent runs once per item, alone: it never sees the other members of the
group, the answer, the level or the generator's specification. The seed sends
the item's `state` and question unchanged. Upstream designed this request to
need no prompt template, and a structured state goes as a JSON object, as
upstream's `runner/bench_http.py` sends it. A non-empty `prompts/instructions.md`
goes in front of the question's own instructions. The seed file is empty, so the
seed request is exactly upstream's.

A candidate may add text to the instructions or option texts, pre-process the
state, make several decision calls, or add chat calls (`session.chat`, traced
and following the run's model). `decide.py` rejects any request that drops,
reorders, adds or edits an option, changes the question type, or removes the
question text.

Every decision call is traced as `model_call(operation="systemone",
provider="typesafe", model=..., server_address="api.typesafe.ai")` with its
input, output and usage. Each case runs in a `decisionpivot.group` stage with
one `decisionpivot.item` stage per item.

## Scoring

An item is right when its top option is the gold option. The top option is the
option with the largest probability, and the first listed wins a tie, as in
upstream `scoring/score.py`. A failed item counts as wrong with the uniform
distribution, as upstream does, and its error is recorded.

| Score | Per group |
|---|---|
| `objective`, `pair_correct` | 1 if every item in the group is right |
| `item_accuracy` | share of items in the group that are right |
| `level3_pair_correct` | pair correct, on level-3 groups only (its mean is level-3 pair accuracy) |
| `min_top_probability` | the lowest top probability in the group |

Each case also has one check per item (expected and predicted option) and
carries `family`, `level` and `layer` as case metadata. Beaker requires field
scores in [0, 1], so the level (1–3) is metadata, not a field score.

Run-level numbers come from upstream, not from this recipe. `evaluate` writes
upstream answer files (`runs/<system>/<family>.jsonl`, `{"id", "p": {...}}`),
runs the pinned `scoring/score.py --all` on them unmodified, and reports its
numbers: accuracy, pair accuracy, level-3 pair accuracy, pair direction,
automation at 5%, accuracy at 0.90 and calibration error (15 bins), per family
and as upstream's mean over families, with its 95% bootstrap intervals.

## Evaluate locally

Set `TYPESAFE_API_KEY` in your shell. This command makes paid API calls:

```bash
uv run decisionpivot evaluate --data /tmp/decisionpivot-data --split validation --limit 100 \
  --output /tmp/decisionpivot-validation
```

Use `--split heldout` with no `--limit` for the held-out evaluation, and
`--prompts <directory>` for a candidate's prompts. To compare another decision
model behind a TypeSafe-compatible endpoint, add `--model`, `--base-url` and
`--api-key-env`. The output directory holds per-group results (`cases.jsonl`),
raw outputs, the upstream answer files and scoring folder, upstream's
`scores/report.md` and `scores.json`, and `summary.json`. The summary adds
p50/p95 latency per decision call and cost per 1,000 decisions at Jev's reported
$0.042 per million input tokens. Any failed item makes the command exit
non-zero.

Run offline tests with `BEAKER_AGENT_KEY= uv run python -m pytest -q`. They make
no paid calls, but they fetch the pinned upstream commit and run `prepare` once
(about a minute).

## Evaluation only

Upstream asks that the items never be used as training data or put into the
prompts of a model that is being trained. This recipe trains no model: it
optimizes the agent's prompts and code, and the decision model only runs
inference. To keep within the spirit of those terms:

- Seven families are optimized on fresh draws; the published items of those
  families are held out.
- No generated item is committed. `prepare` regenerates them.
- `uv run decisionpivot leak-check --data <dir>` fails if any item state, or any
  run of 12 or more consecutive words from a state, appears in `prompts/` or
  `src/decisionpivot/agent.py` (or in the files given with `--path`). CI runs it
  on the committed files. Run it on any optimized candidate before committing
  its prompts or agent code.

**Provider data policies** (checked 25 November 2026):

- **TypeSafe (Jev)** does not train on API inputs. Its
  [privacy policy](https://typesafe.ai/legal/privacy-policy) says it will not
  train or fine-tune models on prompts or other input, and its
  [models page](https://docs.typesafe.ai/models) says Jev is not trained on
  customer requests or responses. No fixed retention period is published for
  standard use; zero data retention is available on request.
- **OpenAI** (including the Decisions endpoint, `POST /v1/decisions`) does not
  train on API data unless the customer opts in. The
  ["Your data" guide](https://developers.openai.com/api/docs/guides/your-data)
  lists `/v1/decisions` with no training use. Abuse-monitoring logs are kept for
  up to 30 days by default; zero data retention needs OpenAI's prior approval.

## Seed baseline

Not yet measured. This needs a TypeSafe API key and paid calls on validation and
held out.

## References

- Zhuang et al. (2026), [*DecisionPivot: A Diagnostic Benchmark for Decision Models in LLM Systems*](https://arxiv.org/abs/2610.28522). Formerly called DecideBench.
- Code: [`StartLuxLabs/DecisionPivot`](https://github.com/StartLuxLabs/DecisionPivot) at `fd57688f033864e64c452561ffdfcba9739d1811`, run unmodified.

See [ATTRIBUTION.md](ATTRIBUTION.md), [LICENSE](LICENSE) and [NOTICE](NOTICE).
