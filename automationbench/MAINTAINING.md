# Maintaining the AutomationBench recipe

For installation, runs, and results, start with the [user guide](README.md).
This guide is for changes to the integration, custom datasets, or provider adapters.

## Code ownership

| Location | Responsibility |
| --- | --- |
| Pinned AutomationBench-PI dependency | Tasks, simulated apps, API catalog, tools, world models, and scalar grading. |
| `src/automationbench_skills/` | Model/tool loop, skills, tracing, provider-message conversion, and grading-state transport. Beaker can edit this code. |
| `src/automationbench_skills/vendored/` | Copied provider clients and routing. Revisions and local adaptations are recorded in [ATTRIBUTION.md](ATTRIBUTION.md). |
| `.beaker/` | Dataset validation and export, Beaker integration, and scoring from supplied assertions and returned state. |

Each rollout owns its world and MCP session. The adapter accesses PI's private
`_world` and `_api` attributes. Review these uses when updating the dependency pin.

Beaker runs candidate code separately from the controller's loaded scoring
integration. Preserve that separation. The scorer must use the supplied case's
assertions and initial state, not candidate-provided scores or assertions.

PI returns a scalar score. The CLI and hosted scorer reconstruct per-assertion
details separately. Keep both aligned with PI's exclusion rules; do not load
candidate code into the scoring process to share those calculations.

## Grading state

`run_case` returns the serialized final world in `CaseResult.context.end_state`.
The scorer rebuilds a `WorldState` from it and calls PI's `partial_credit` rubric.

PI also tracks Google Sheets writes in `_updated_row_keys`, which is absent from
`WorldState.model_dump()`. `grading_state.py` captures this evidence separately.
The integration transports it in `context.grading_state` and restores it before
scoring. Preserve actual writes, including writes later reverted to initial values;
final cell values cannot reconstruct this history.

Beaker Checks show assertion conditions and available grading observations.
Diagnostics are partial evidence, not a complete causal explanation. OUTPUT shows
app changes once per task. `_budget_output` reduces display content near the
transport limit while preserving the full scoring context.

## Custom datasets

Use `.beaker/upload_splits.py` for the standard quickstart dataset. For custom
rows, use `.beaker/case_fixture.py` as the schema and serialization authority.
Each row contains:

| Field | Contents |
| --- | --- |
| `id` | A unique case ID across train and test. |
| `input.task_name` | The fixture's task name. |
| `expected.assertions` | The fixture's assertion list. |
| `runtime_fixture.task` | `example_id`, `prompt`, `answer`, and `info`. |
| `runtime_fixture.task.info` | `task_name`, `initial_state`, `assertions`, `zapier_tools`, and any upstream metadata. |
| `runtime_fixture.scenario` | Source, fallback time, tool fault, and trace-capture settings. |
| `metadata` | Source revision and task/scenario hashes. |

Construct a `CaseFixture`, serialize it with `model_dump(mode="json")`, and use
its `hashes()` method for `task_sha256` and `scenario_sha256`. Do not duplicate the
hashing algorithm in an exporter. The loader checks task names, assertions,
source revisions, hashes, world-schema compatibility, and assertion types.

`scenario.source` is `pi` or `original`; `metadata.benchmark_commit` must match
`SOURCE_COMMITS` for that source. These fields record provenance. All fixtures
execute under the pinned PI tools and graders. Task contents come from the row,
not from a name-based lookup.

For deliberate repeated cases, use split-qualified IDs such as `train:case-123`
and `test:case-123`. A shared `metadata.reference_id` can link them to external
labels. Such copies are not independent examples and are not a held-out test set.
Keep reference labels outside model inputs.

### Controlled evaluation faults

The scenario supports two tool faults for data-quality experiments:

- `empty_email_lookup`: replace a successful Gmail list result with an empty list.
- `drop_opportunity_amount`: omit the amount from a Salesforce opportunity update.

Neither changes task data or graders. `trace_capture: incomplete` omits alternating
tool spans while retaining execution, model messages, state, and grading. This
simulates missing trace evidence. Validate labels under the same tool interface
and fault settings used for the evaluation.

## Provider behavior

Hosted runs use Chat Completions or Anthropic Messages clients with Beaker tracing.
The local CLI selects Responses for direct OpenAI models, Anthropic Messages for
Claude, Gemini interactions for Gemini, and Chat Completions for OpenRouter.
Explicit API and gateway settings override automatic routing.

The recipe's Chat Completions client uses a 24,576-token reasoning allowance plus
8,192 answer tokens, up to three attempts, a 300-second request timeout, and a
600-second turn deadline. SDK retries are disabled for that client. Keep response
validation shared with the retry loop so refusals and reasoning signals are not
mistaken for empty replies. Count discarded responses in usage and cost totals.

For Python callers, `ModelSpec.extra_body` can set a supported reasoning token
budget or a lower output cap. The CLI does not expose `extra_body`. A reasoning
budget applies to direct OpenRouter calls; the hosted Beaker gateway does not
accept that token-budget setting. Keep these controls aligned with provider support.

Responses retries empty output up to three attempts. Other native provider retry
behavior follows the copied clients. Keep vendored changes narrow and record them
in ATTRIBUTION.md so source comparisons remain useful.

Rollout errors remain exception objects until the integration classifies them.
Hosted provider failures request a case retry. Context overflow and task deadlines
retain earned credit. Reasoning that exhausts the Chat Completions output cap
without an answer raises a model error. Local runs score the world left by the
agent; inspect the error field alongside the score.

## Development checks

Run the recipe's commands from `.github/workflows/checks.yml` in the
`automationbench` directory:

```bash
uv sync --group dev --locked
uv run ruff check
uv run ruff format --check
uv run python -m mypy
uv build --no-sources
BEAKER_AGENT_KEY= uv run python -m pytest -q
```

The tests use scripted or mocked clients. They cover dataset validation, grading
across the JSON boundary, provider errors, retries, and usage accounting. They do
not establish live model quality. Ruff and mypy exclude the vendored subtree.

After changing the dependency pin, verify world serialization, private API access,
assertion exclusions, spreadsheet write history, and provider-message replay.
Rebuild the hosted integration image before evaluating a changed recipe.
