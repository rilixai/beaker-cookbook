# ALFWorld decision agent

A household agent on [ALFWorld](https://arxiv.org/abs/2010.03768) that picks
every action with a decision model instead of a chat model. ALFWorld's text
games ("put a clean mug in the coffee machine") list the legal actions at every
step, so the agent is a loop of Choice questions with no chat call: one native
`jev-1.14.0` TypeSafe request per step, until the task is done or 50 steps run
out.

Beaker optimizes `prompts/` and `src/alfworld_agent/agent.py`. The game files,
the environment and its rules (`env.py`), every model call (`decide.py`), the
data (`data.py`) and scoring (`scoring.py`) stay fixed.

## Run with Beaker

From this directory, with Python 3.12 and uv:

```bash
uv sync --group dev --locked
uv run alfworld-agent prepare --output /tmp/alfworld-data
uv run beaker agent setup "ALFWorld" --integration-id alfworld
uv run beaker dataset upload /tmp/alfworld-data/optimization --name alfworld --split train=3553 --split test=140

# Replace <revision> with the revision returned by upload.
uv run beaker run smoke --strict --integration-id alfworld --dataset 'alfworld@<revision>'
uv run beaker run trigger --integration-id alfworld --dataset 'alfworld@<revision>'
```

For a cheap trial run first, build a small dataset with the first games of each
task type (2 per type: 12 games per split):

```bash
uv run alfworld-agent prepare --output /tmp/alfworld-small --games-per-type 2
uv run beaker dataset upload /tmp/alfworld-small/optimization --name alfworld-small \
  --split train=12 --split test=12
```

Upload only `optimization/`: its `test.jsonl` is validation data. Keep
`heldout/test.jsonl` for final evaluation of a frozen candidate.

Without a model choice, decision calls go to native Jev. A run with a model
choice sends them to that model through Beaker's gateway. Smoke checks
structure without model calls.

The hosted environment installs this package, including TextWorld's PDDL
engine, which PyPI ships only as source, so `.beaker/beaker.yaml` installs
`cmake`, `g++` and `make`. The sandbox downloads and verifies the 36 MB game
archive once, in `prepare_run`.

## Data

`prepare` downloads three pinned ALFWorld release files, checks their SHA-256
(see [ATTRIBUTION.md](ATTRIBUTION.md)), and works offline after the first
fetch. It then runs upstream's own filter, `AlfredTWEnv.collect_game_files`
(unmodified), and fails if the game counts differ from the pinned ones. One
case is one game: a full episode.

| File | Upstream split | Games |
|---|---|---:|
| `optimization/train.jsonl` | `train` | 3,553 |
| `optimization/test.jsonl` (validation) | `valid_seen` | 140 |
| `heldout/test.jsonl` (held out) | `valid_unseen` | 134 |

`valid_unseen` is the split LLM-agent papers report, so held-out numbers can be
compared with published results. Upstream's `valid_train` (200 games) is not
used.

| Task type | Train | Validation | Held out |
|---|---:|---:|---:|
| `pick_and_place_simple` | 790 | 35 | 24 |
| `look_at_obj_in_light` | 308 | 13 | 18 |
| `pick_clean_then_place_in_recep` | 650 | 27 | 31 |
| `pick_heat_then_place_in_recep` | 459 | 16 | 23 |
| `pick_cool_then_place_in_recep` | 533 | 25 | 21 |
| `pick_two_obj_and_place` | 813 | 24 | 17 |

Each file is ordered by a seeded hash (seed `alfworld-v1`), stratified by task
type, so any prefix is a fair sample. `manifest.json` records the package
versions, source URLs and checksums, the seed, and counts per split and task
type.

Each row is
`{"id": "alfworld-<split>-<hash of game path>", "input": {"split", "game"}, "expected": {"task_type"}}`.
`game` is a path relative to the extracted archive. The task type comes from
the game's folder name and is used only for reporting.

## The agent

Each step, `env.py` gives the agent a `View`: the task goal, the current
observation, the history so far (actions and observations) and the admissible
commands. The seed asks one Choice question:

- state: the goal, the last 10 steps and the current observation
- instructions: `prompts/instructions.md`
- options: every admissible command, its text as both key and description

and returns the chosen command, which `env.py` then plays.

Rules, enforced outside the editable files:

- The command sent to the game must be currently admissible. Anything else
  fails the case; it is never a silent no-op.
- The agent never sees `won`, the game file path, PDDL facts, the expert plan
  or the task type. `env.py` requests only `admissible_commands` and `won`
  from TextWorld and keeps `won` to itself.
- Every decision request offers only admissible commands: at least 2, at most
  255. A request may leave commands out, write its own option descriptions, and
  the agent may ask more than one question per step.
- The agent may keep its own memory across steps; a new `Agent` is made for
  each episode.
- An episode stops at 50 steps.

Across 674 games (all of `valid_seen` and `valid_unseen` and 400 training games,
replaying the walkthrough and then 50 random actions), a step had at most 73
admissible commands (p99 58, median 27), well under the 255-option cap.

## Models and tracing

Every decision call goes through `decide.py` and is traced as a model call with
operation `systemone`, provider `typesafe`, the model and server
`api.typesafe.ai`, with its input, output and token usage. The model name is
one constant, `decide.MODEL`. If a run selects a model, decision calls go to it
through `inference_target(runtime)`.

A candidate may add chat calls with `session.chat`. They are traced as chat
model calls and follow the run's model; without one they use `gpt-6-luna`.

Each case is a trace stage `alfworld.episode` (input: case ID) with one child
stage `alfworld.step` per step (inputs: step number and option count). The full
transcript (each step's observation, options, probabilities and chosen action)
is in the result context. The displayed output is the goal, the last three
steps and the outcome.

## Scoring

The objective is 1 if the game is won within 50 steps, else 0. ALFWorld's text
env reports no partial progress (its score stays 0 until the game is won), so
there is no partial credit.

| Field score | Meaning |
|---|---|
| `success` | Same as the objective |
| `steps` | Steps used ÷ 50 |
| `decision_calls` | Decision calls ÷ 100, capped at 1 |
| `repeat_rate` | Share of steps that repeat the previous step's action and observation |
| `mean_top_probability` | Mean probability of the top option across decision calls |

Beaker field scores must lie in [0, 1], so steps and calls are scaled. The raw
counts are in the output. Checks: `success`, and `step_limit`, which fails when
the episode used all 50 steps without winning. A game error, a decision-API
error, a rejected request or an inadmissible command fails the case with its
error recorded.

## Evaluate locally

Set `TYPESAFE_API_KEY` in your shell; these commands make paid API calls:

```bash
uv run alfworld-agent evaluate --data /tmp/alfworld-data --split validation --limit 20 \
  --output /tmp/alfworld-validation.jsonl
uv run alfworld-agent play --data /tmp/alfworld-data --game alfworld-valid_seen-5617f4e0f84a
```

Use `--split test` with no `--limit` for the held-out evaluation and
`--prompts <directory>` for a candidate's prompts. `evaluate` writes one result
per episode (with its transcript) and `<output>.summary.json`. The summary has
the success rate overall and per task type, mean steps on successes, decision
calls per episode, repeat rate, p50/p95 latency per decision, and cost per
episode at Jev's reported $0.042 per million input tokens. Any failed episode
makes it exit non-zero. `play` prints one episode step by step.

Run offline tests with `BEAKER_AGENT_KEY= uv run python -m pytest -q`. They
download the 36 MB game archive on first use and make no paid calls.

The text-only install works on macOS arm64 with Python 3.12; TextWorld and
Fast Downward build from source there, which needs the Xcode command-line
tools.

## Seed baseline

Not yet measured: this needs a TypeSafe API key and paid calls on validation
(3 runs) and held out.

## References

- Shridhar et al. (ICLR 2021), [*ALFWorld: Aligning Text and Embodied Environments for Interactive Learning*](https://arxiv.org/abs/2010.03768).
- Côté et al. (2018), [*TextWorld: A Learning Environment for Text-based Games*](https://arxiv.org/abs/1806.11532).
- Shridhar et al. (CVPR 2020), [*ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks*](https://arxiv.org/abs/1912.01734).

See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).
