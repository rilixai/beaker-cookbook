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
from rilixai.adapters import BaseCaseRunner, AttributeApplier
from rilixai.metrics import FieldConfig
from rilixai.prompt_optimization import Case

from .agent import MyAgent, MyRecord, MyOutput


@spec(
    name="my-agent",
    field_configs=[FieldConfig(name="answer", comparators="exact_match")],
    task_type="my-agent",
)
class MyAgentRunner(BaseCaseRunner[MyRecord, MyOutput]):
    def __init__(self, ctx):
        self.agent = MyAgent()
        super().__init__(applier=AttributeApplier(target=self.agent, mapping={"system": "system_prompt"}))

    async def run_case(self, record):
        return await self.agent.run(record)

    def cases_by_split(self, ctx):
        return {"train": [...], "validation": [...]}  # list[Case]
```

---

## 1. What rilixai expects from your agent

Exactly two things:

1. **An async callable that takes one record and returns one output** — your
   `run_case(record)` method. The record is whatever you put in
   `Case.input`; the output is what gets scored.
2. **A way to receive the candidate's prompt components** — a `dict[str, str]`
   rilixai hands you each evaluation pass. You expose that through a
   `ComponentApplier` (§3): an attribute setter, deps injection, or a custom
   callable.

That's the entire surface. rilixai never inspects your agent internals, your
tool definitions, your dataset format, or your scoring math. It only touches
the components dict and your `run_case` callable.

---

## 2. The `PromptCandidate.components` schema

`components` is a `dict[str, str]` — keys are whatever **you** name them, values
are the prompt strings the optimizer rewrites. The reflection layer treats each
key as a component to optimize independently. Whatever you name a key
(`"policy_prompt"`, `"system"`, `"summarize"`) is what the reflection LM sees
and what your feedback (§6) keys off.

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

`FieldConfig.extract_from` (§5) resolves through dotted access — attributes
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
comparator — register one as a custom comparator on your metrics class (§5).
The wrapper above still applies: name the field; the judge does the comparison.
apex_agents does exactly this — see `apex_agents/metrics.py`
(`score_rubric` / `build_rubric_judge`) and `ApexAgentsMetrics` in
`apex_agents/rilixai_spec.py`.

---

## 3. Picking a `ComponentApplier` — framework cheatsheet

Almost everything maps to `AttributeApplier` with a different attribute name.
Only PydanticAI's deps pattern needs its own applier.

| Agent framework | Applier | Concrete wiring |
|---|---|---|
| PydanticAI (prompts via deps) | `PydanticAIDepsApplier` | `PydanticAIDepsApplier(deps_factory=lambda c: MyDeps(**c), holder=holder, seed_reader=lambda: {...})` |
| PydanticAI (prompts via attrs) | `AttributeApplier` | `AttributeApplier(target=agent, mapping={"system": "system_prompt"})` |
| Raw OpenAI Chat Completions | `AttributeApplier` | Hold the system string on a small wrapper class; `AttributeApplier(target=wrapper, mapping={"system": "system_prompt"})` |
| Anthropic Python SDK | `AttributeApplier` | Same pattern as OpenAI Chat — wrap your `system=` string on a holder. |
| OpenAI Agents SDK | `AttributeApplier` | `AttributeApplier(target=agent, mapping={"system": "instructions"})` |
| Claude Agent SDK | `AttributeApplier` | `AttributeApplier(target=agent, mapping={"system": "instructions"})` |
| LangChain Runnable | `AttributeApplier` | Hold the prompt template on a wrapper; map into its `template` attribute. |
| Anything else | `CallableApplier(apply=..., read=...)` | One- or two-callable escape hatch. |

The pattern is uniform enough that "switch frameworks" usually means "change two
strings in your `AttributeApplier(mapping=...)` call."

Every applier is symmetric: `apply(components)` pushes prompts in, and `read()`
returns the agent's current prompts back out — rilixai calls `read()` once at
spec-build time to capture your **seed candidate** automatically, so you don't
author one by hand. (`AttributeApplier` reads/writes the same attributes;
`CallableApplier` / `PydanticAIDepsApplier` take a `read` / `seed_reader`
callable.)

Both cookbook recipes use `AttributeApplier` directly. The agent exposes normal
prompt attributes, and `rilixai_spec.py` is the single place that declares the
rilixai component names and maps them onto those attributes:

```python
SYSTEM = "system_prompt"
SUMMARY = "summarize_prompt"


super().__init__(
    applier=AttributeApplier(
        target=self.agent,
        mapping={
            SYSTEM: "system_prompt",
            SUMMARY: "summarize_prompt",
        },
    )
)
```

If changing a prompt needs side effects, make the attribute a property on your
agent. For example, the HotpotQA recipe uses a `policy_prompt` setter to rebuild
its inner PydanticAI `Agent` because that framework bakes `system_prompt` into
the agent at construction time.

---

## 4. Loading your data

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

`ctx.config` (§7) carries the run's knobs (split sizes, model, etc.), so size
your splits from it: `cases[: ctx.config.train_size]`.

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

## 5. Scoring: the `FieldConfig` cookbook

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
    fields = [
        FieldConfig(name="answer", comparators="hotpot_exact_match", weight=1.0),
        FieldConfig(name="answer_f1", extract_from="answer", comparators="hotpot_f1", weight=0.0),
        FieldConfig(name="titles_recall", extract_from="retrieved_titles",
                    compare_to="supporting_titles", comparators="set_recall", weight=0.0),
    ]
    comparators = {"hotpot_exact_match": _em, "hotpot_f1": _f1}  # your callables
```

A comparator is just `(predicted, expected) -> float in [0, 1]`. For LLM-judge
scoring, your comparator can ignore `expected` and read a precomputed score off
the output (apex_agents runs the judge in the runner and reads it back this
way).

---

## 6. Custom feedback narratives — when to bother

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
    @per_component_feedback("system")
    def system(self, case, output) -> str:
        return f"On case {case.case_id} the agent answered {output.answer!r}; " \
               f"expected {case.ground_truth['answer']!r}. ..."
```

Custom narratives typically improve convergence by 20–40% on tasks with rich
domain signal. Add them incrementally — one component at a time. See
`hotpotqa/feedback.py` and `apex_agents/feedback.py` for full examples.

---

## 7. End-to-end: from zero to a queued run

1. **Scaffold** (optional): `uv run rilixai init spec --name my-agent
   --from-agent ./my_agent/agent.py` writes a `rilixai_spec.py` skeleton with
   the applier + component names pre-detected from your agent file. Or copy a
   cookbook recipe.
2. **Fill in** `agent.py` (your production agent), the `FieldConfig`s, and
   `run_case` / `cases_by_split` in `rilixai_spec.py`.
3. **Verify** the registration test passes:
   `uv run python -m pytest <member>/tests` — the one-line
   `assert_spec_registered("my-agent")` confirms `@spec` discovery works.
4. **Dry-run one case locally.** `uv run rilixai dry-run --config '{...}'`
   builds the spec, applies the seed prompts, runs one case, and prints scores
   before you spend hosted optimization budget.
5. **Push + run.** Set `RILIXAI_API_KEY` / `RILIXAI_API_BASE_URL`, then
   `uv run rilixai push --member <member>` to build + promote the image and
   `uv run rilixai trigger` to queue a run (see each recipe's README for the
   exact commands). `OPENAI_API_KEY` and any other provider keys are bound as
   project-level secrets on rilixai's side, injected into each sandbox.

`ctx.config` inside the sandbox is validated against the Pydantic schema you
pass to `@spec(config_schema=...)`, so a typo'd trigger key fails fast with a
clear error instead of silently using a default.
