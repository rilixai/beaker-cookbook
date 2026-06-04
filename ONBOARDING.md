# Onboarding your agent to rilixai

You have a production agent. You want rilixai's optimizer to improve its
prompts. This doc is the whole contract — read it once and you'll know exactly
what to write.

The short version: your integration is **one `@spec`-decorated
[`BaseCaseRunner`](https://github.com/rilixai/rilixai) subclass** in a file
conventionally named `rilixai_spec.py`. rilixai assembles everything else
(scoring, the seed candidate, per-component feedback) from what you declare on
the class. The two cookbook recipes — `hotpotqa/rilixai_spec.py` and
`apex_agents/rilixai_spec.py` — are working references for everything below.

```python
from rilixai import spec
from rilixai.adapters import BaseCaseRunner
from rilixai.metrics import FieldConfig
from rilixai.prompt_optimization import Case

from .agent import DEFAULT_SYSTEM_PROMPT, MyAgent, MyRecord, MyOutput


@spec(
    name="my-agent",
    field_configs=[FieldConfig(name="answer", comparators="exact_match")],
    task_type="my-agent",
)
class MyAgentRunner(BaseCaseRunner[MyRecord, MyOutput]):
    def __init__(self, ctx):
        super().__init__(prompts={"system_prompt": DEFAULT_SYSTEM_PROMPT})

    async def run_case(self, record):
        agent = MyAgent(system_prompt=self.prompt("system_prompt"))
        return await agent.run(record)

    def cases_by_split(self, ctx):
        return {"train": [...], "validation": [...]}  # list[Case]
```

---

## 1. What rilixai expects from your agent

Exactly two things:

1. **An async callable that takes one record and returns one output** — your
   `run_case(record)` method. The record is whatever you put in
   `Case.input`; the output is what gets scored.
2. **The prompts rilixai may optimize** — a `dict[str, str]` of prompt name →
   default prompt string. During a run, use `self.prompt("<name>")` wherever
   your production agent needs that prompt string.

That's the entire surface. rilixai never inspects your agent internals, your
tool definitions, your dataset format, or your scoring math. It only touches
the prompt-component dict and your `run_case` callable.

---

## 2. The `PromptCandidate.components` schema

`components` is a `dict[str, str]` — keys are whatever **you** name them, values
are the prompt strings the optimizer rewrites. The reflection layer treats each
key as a component to optimize independently.

Prefer naming each component after the prompt a developer would recognize in
your agent: `"system_prompt"`, `"policy_prompt"`, `"summarize_prompt"`,
`"task_template"`. Whatever you name a key is what the reflection LM sees,
what `self.prompt("<name>")` reads, and what your feedback (§9) keys off.

### Output format expectations

Your `run_case(record)` must return one of:

| Output type | Field access | Supported? |
|---|---|---|
| Pydantic `BaseModel` | `output.field` (attribute) | ✅ recommended |
| Dataclass | `output.field` (attribute) | ✅ |
| `dict[str, Any]` | `output["field"]` (item) | ✅ |
| NamedTuple / TypedDict | attribute or item | ✅ |
| Nested combinations | `output.foo.bar` / `output["foo"]["bar"]` | ✅ |
| Raw `str` | n/a | ❌ — wrap it (see below) |
| Raw scalar (int, float, bool) | n/a | ❌ — wrap it |

`FieldConfig.extract_from` (§6) resolves through dotted access — attributes
first, items second. If your agent currently returns a bare string, wrap it:

```python
# Don't:
async def run_case(self, record):
    return await self.agent.run(record.question)  # returns str

# Do:
class Output(BaseModel):
    answer: str

async def run_case(self, record):
    answer = await self.agent.run(record.question)
    return Output(answer=answer)
```

**Why?** Field-level scoring needs *named* outputs to map into
`FieldConfig.name`. A bare string has nothing to address. The wrap is two lines
and makes the output self-documenting.

**Free-text outputs that can't be compared by string equality** (open-ended
generation, summaries, multi-paragraph answers) are scored with an LLM-judge
comparator — register one as a custom comparator on your metrics class (§6).
The wrapper above still applies: name the field; the judge does the comparison.
apex_agents does exactly this — see `apex_agents/metrics.py`
(`score_rubric` / `build_rubric_judge`) and `ApexAgentsMetrics` in
`apex_agents/rilixai_spec.py`.

---

## 3. Declaring prompts — the default path

Most integrations do not need an applier. Put the prompt defaults in the runner
constructor and read the active candidate prompt by name inside `run_case`:

```python
class MyRunner(BaseCaseRunner[MyRecord, MyOutput]):
    def __init__(self, ctx):
        super().__init__(
            prompts={
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "summarize_prompt": DEFAULT_SUMMARIZE_PROMPT,
            }
        )

    async def run_case(self, record):
        agent = MyAgent(
            system_prompt=self.prompt("system_prompt"),
            summarize_prompt=self.prompt("summarize_prompt"),
        )
        return await agent.forward(record)
```

That is the common shape:

- `prompts={...}` declares the seed candidate and the component names.
- `self.prompt("system_prompt")` returns the active value for this case.
- Concurrent case execution is safe because the active candidate prompts are
  stored per invocation, not on shared mutable runner state.

Hardcoding the prompt names at the call site is fine. It makes the integration
obvious: "this is where the optimized `system_prompt` enters my agent." Use
constants only if the same name is repeated enough that the indirection pays
for itself.

## 4. When to use appliers

Appliers are extra support for production agents that cannot simply receive
prompt strings through a constructor or `run(...)` call. They are escape
hatches for existing framework objects, not the default onboarding path.

Use an applier when:

- You already have a persistent agent object and the framework stores prompts
  on mutable attributes, such as `agent.instructions`.
- The prompt lives at a nested path, such as `chain.prompt.template`.
- Applying a candidate requires custom side effects, such as rebuilding a
  cached chain, recreating deps, or calling your agent's existing setter.
- Your production code already exposes a prompt-setting API and you want to
  keep the runner thin.

The applier options are:

| Need | Support | Example |
|---|---|---|
| Mutate prompt attributes | `AttributeApplier` | `AttributeApplier(target=agent, components=("system_prompt",))` |
| Public component name differs from storage path | `AttributeApplier(mapping=...)` | `AttributeApplier(target=agent, mapping={"system_prompt": "instructions"})` |
| Nested prompt path | `AttributeApplier(mapping=...)` | `AttributeApplier(target=chain, mapping={"summary_prompt": "prompt.template"})` |
| Immutable or recreated deps object | `PydanticAIDepsApplier` | `PydanticAIDepsApplier(deps_factory=lambda c: MyDeps(**c), holder=holder, seed_reader=lambda: {...})` |
| Arbitrary apply/read behavior | `CallableApplier` | `CallableApplier(apply=set_prompts, read=current_prompts)` |

Every applier is symmetric: `apply(components)` pushes prompts in, and `read()`
returns the current prompts back out. rilixai calls `read()` once at spec-build
time to capture your seed candidate automatically. If the prompt defaults are
easy to spell as a dict, prefer `super().__init__(prompts={...})`; if prompt
storage is already owned by a production object, use an applier.

Two other prompt-support paths are available:

- If your prompt defaults are generated or loaded from a place the runner should
  not read, pass an explicit seed with `@spec(seed={...})`.
- `rilixai init spec --from-agent ./agent.py` can inspect source code to
  pre-fill a scaffold, but runtime prompt discovery is intentionally explicit:
  your runner declares the component names it wants rilixai to optimize.

---

## 5. Loading your data

rilixai does **not** ship a data loader — formats vary too much (HuggingFace,
JSONL, Postgres, pandas, an internal eval harness) for a base class to be
net-positive. You own `cases_by_split(ctx) -> dict[str, list[Case]]`. A
`Case` is:

```python
from rilixai.prompt_optimization import Case

Case(
    input=record,                       # your domain object — passed to run_case
    case_id="row-42",                 # stable id
    ground_truth={"answer": "Paris"},   # dict the metrics compare against
    group_key="default",                # optional; used by the failure-focused sampler
)
```

Most production teams start with a local JSONL eval set or an immutable hosted
dataset artifact. rilixai ships small helpers for those repetitive pieces while
leaving your domain row → record mapping in your code.

Three common shapes:

```python
# (a) HuggingFace — see hotpotqa/data/dataset.py for the full version
def cases_by_split(self, ctx):
    from datasets import load_dataset
    rows = load_dataset("my-org/my-bench", split="train")
    cases = [
        Case(input=_to_record(r), case_id=r["id"], ground_truth={"answer": r["answer"]})
        for r in rows
    ]
    return {"train": cases[:150], "validation": cases[150:250]}

# (b) JSONL on disk — the common "I have an eval file" path
def cases_by_split(self, ctx):
    from pathlib import Path
    from rilixai.data import load_jsonl_cases

    return load_jsonl_cases(
        Path("data"),
        {"train": "train.jsonl", "validation": "validation.jsonl"},
        row_to_case=lambda row, split: Case(
            input=_to_record(row),
            case_id=str(row["id"]),
            ground_truth=row["gt"],
            group_key=row.get("customer_id") or "default",
        ),
    )

# (c) In-memory from a pandas DataFrame
def cases_by_split(self, ctx):
    df = my_dataframe()
    cases = [
        Case(input=row.to_dict(), case_id=str(i), ground_truth={"label": row.label})
        for i, row in df.iterrows()
    ]
    return {"train": cases, "validation": cases}
```

`ctx.config` carries the run's knobs (split sizes, model, etc.), so size your
splits from it: `cases[: ctx.config.train_size]`.

For hosted production runs where your app snapshots eval data before queuing a
run, attach the snapshot as an input artifact and materialize it inside the
runner:

```python
import tempfile
from pathlib import Path

from rilixai.data import (
    find_dataset_artifact,
    load_jsonl_cases,
    materialize_dataset_artifact,
)


def cases_by_split(self, ctx):
    artifact = find_dataset_artifact(ctx, kinds=["dataset_snapshot", "my_agent_dataset"])
    split_files = {"train": "train.jsonl", "validation": "validation.jsonl"}
    with tempfile.TemporaryDirectory(prefix="my-agent-dataset-") as tmpdir:
        dataset_dir = materialize_dataset_artifact(
            artifact,
            Path(tmpdir) / "dataset",
            required_files=tuple(split_files.values()),  # needed for s3:// artifacts
        )
        return load_jsonl_cases(dataset_dir, split_files, row_to_case=_row_to_case)
```

That is the same pattern production integrations use for immutable S3 snapshots:
your app owns the snapshot; rilixai owns applying candidates, running cases, and
recording optimized prompts.

---

## 6. Scoring: the `FieldConfig` cookbook

Declare what to score by listing `FieldConfig`s — either inline on `@spec(...)`
or on a `BaseMetricsCalculator` subclass. The `name` doubles as the default
dotted path on both the output and the ground-truth dict.

```python
from rilixai.metrics import FieldConfig

# Single score on a single field (the 80% case)
FieldConfig(name="answer", comparators="exact_match")

# Multiple scores on one field → emits answer_exact_match + answer_f1_score
FieldConfig(name="answer", comparators=["exact_match", "f1_score"])

# Result path differs from ground-truth path
FieldConfig(
    name="titles_recall",
    extract_from="retrieved_titles",     # path on your output
    compare_to="supporting_titles",      # path on ground_truth
    comparators="set_recall",
)

# Weighted: only this field drives candidate selection
FieldConfig(name="answer", comparators="exact_match", weight=1.0)
FieldConfig(name="answer_f1", extract_from="answer", comparators="f1_score", weight=0.0)
```

Shipped comparators (autocompleted via the `ComparatorName` literal):
`exact_match`, `f1_score`, `set_recall`, `set_f1`, `numeric_close`, `llm_judge`.

**Custom comparators** register on a `BaseMetricsCalculator` subclass — pass the
class to `@spec(field_configs=...)`. This is how both recipes reuse
domain-specific scorers:

```python
from rilixai.metrics import BaseMetricsCalculator, FieldConfig

class HotpotQAMetrics(BaseMetricsCalculator):
    # Use shipped comparator names directly when they fit:
    # exact_match, f1_score, set_recall, set_f1, numeric_close, llm_judge.
    # Register custom names in `comparators` only for domain-specific scoring.
    fields = [
        FieldConfig(name="answer", comparators="exact_match", weight=1.0),
        FieldConfig(name="answer_f1", extract_from="answer", comparators="hotpot_f1", weight=0.0),
        FieldConfig(name="titles_recall", extract_from="retrieved_titles",
                    compare_to="supporting_titles", comparators="set_recall", weight=0.0),
    ]
    # HotpotQA keeps a custom F1 because the official benchmark treats
    # yes/no/noanswer mismatches differently from rilixai's generic f1_score.
    comparators = {"hotpot_f1": _hotpot_f1}  # only where shipped scorers diverge
```

A comparator is just `(predicted, expected) -> float in [0, 1]`.

---

## 7. LLM-as-judge scoring

Use rilixai's shipped `llm_judge` comparator when the output is too open-ended
for exact string or token matching. Declare it like any other comparator, then
add `judge_config` on the metrics class:

```python
from rilixai.metrics import BaseMetricsCalculator, FieldConfig, LLMJudgeConfig

class MyMetrics(BaseMetricsCalculator):
    fields = [
        FieldConfig(
            name="rubric_pass_rate",
            extract_from="answer",      # path on your output
            compare_to="judge",         # path on Case.ground_truth
            comparators="llm_judge",
            weight=1.0,
        )
    ]
    judge_config = LLMJudgeConfig(
        model="gpt-4.1",
        template="rubric",              # or "open_ended"
        rubric_path="rubric",           # relative to ground_truth["judge"]
        task_prompt_path="prompt",      # relative to ground_truth["judge"]
    )
```

For that example, each case's `ground_truth` should carry the judge bundle:

```python
Case(
    input=record,
    case_id=record.id,
    ground_truth={
        "judge": {
            "prompt": record.prompt,
            "rubric": [{"criteria": "The answer states an enterprise value."}],
        }
    },
)
```

`llm_judge` returns the fraction of rubric criteria marked `MET`. If you already
have a domain judge inside your application, you can also run it in `run_case`
and expose the numeric result as an output field; APEX uses that advanced
pattern so rubric judging can share its benchmark-specific parser.

---

## 8. Custom metrics with default feedback

Metrics and feedback are separate knobs. If you need custom scoring but are
happy with rilixai's templated feedback, declare `field_configs` and omit
`feedback=`:

```python
@spec(
    name="my-agent",
    field_configs=MyMetrics,  # BaseMetricsCalculator or list[FieldConfig]
)
class MyRunner(BaseCaseRunner[MyRecord, MyOutput]):
    ...
```

That path uses your metrics for scoring and attaches `GenericFeedback` to each
declared prompt component. Add `feedback=MyFeedback` only when you want
component-specific narratives. The APEX recipe leaves the custom
`ApexAgentsFeedback` switch commented out to show both paths.

---

## 9. Custom feedback narratives — when to bother

The reflection LM rewrites prompts better when it sees *why* a case scored the
way it did. By default you get `GenericFeedback`, which builds a narrative from
the standard signals (metric scores, predicted-vs-expected, tool-call trace,
errors). **It's optional** — you can ship with zero feedback code and the
optimizer still works.

When the generic template isn't enough, add a feedback class and wire it via
`@spec(feedback=...)`. Each component gets a method decorated with
`@per_component_feedback("<component-name>")`:

```python
from rilixai.adapters import per_component_feedback

class MyFeedback:
    @per_component_feedback("system_prompt")
    def system_prompt(self, case, output) -> str:
        return f"On case {case.case_id} the agent answered {output.answer!r}; " \
               f"expected {case.ground_truth['answer']!r}. ..."
```

Custom narratives typically improve convergence by 20–40% on tasks with rich
domain signal. Add them incrementally — one component at a time. See
`hotpotqa/feedback.py` and `apex_agents/feedback.py` for full examples.

---

## 10. End-to-end: from zero to a queued run

1. **Scaffold** (optional): `uv run rilixai init spec --name my-agent
   --from-agent ./my_agent/agent.py` writes a `rilixai_spec.py` skeleton with
   prompt component names pre-detected from your agent file, plus an applier
   if the inspected agent needs one. Or copy a cookbook recipe.
2. **Fill in** `agent.py` (your production agent), the `FieldConfig`s, and
   `run_case` / `cases_by_split` in `rilixai_spec.py`.
3. **Verify** the registration test passes:
   `uv run python -m pytest <member>/tests` — the one-line
   `assert_spec_registered("my-agent")` confirms `@spec` discovery works.
4. **Dry-run one case locally.** `uv run rilixai dry-run --config '{...}'`
   builds the spec, applies the seed prompts, runs one case, and prints scores
   before you spend hosted optimization budget.
5. **Push + run.** Set `RILIXAI_API_KEY` / `RILIXAI_API_BASE_URL`, then
   `uv run rilixai push --member <member> --version v$(git rev-parse --short HEAD)`
   to build + promote the image and `uv run rilixai trigger` to queue a run
   (see each recipe's README for the exact commands). `OPENAI_API_KEY` and any
   other provider keys are bound as project-level secrets on rilixai's side,
   injected into each sandbox.

`ctx.config` inside the sandbox is validated against the Pydantic schema you
pass to `@spec(config_schema=...)`, so a typo'd trigger key fails fast with a
clear error instead of silently using a default.
