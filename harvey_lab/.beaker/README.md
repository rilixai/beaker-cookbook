# Harvey LAB with Beaker

Run these commands from `harvey_lab/` after `uv sync --group dev --locked`.
This integration uses Beaker SDK 0.6.3 and the same Stirrup agent, prompts,
document extraction, and batched rubric as the standalone recipe.

## Integration contract

- Configuration: `.beaker/beaker.yaml`, integration ID `harvey-lab`.
- Beaker target: the existing `harvey-lab-agent` in `rilixai/beaker-cookbook`.
  For another repository or organization, use `beaker agent setup` and update
  the recorded key before uploading data.
- Editable files: `src/harvey_lab/agent/`, including prompts and the agent
  harness. Dataset loading, model defaults, rubric scoring, and `.beaker/`
  are outside that scope.
- Objective: `criterion_pass_rate`. `all_pass` is also reported, but does not
  contribute to the objective. Every criterion produces a pass/fail check
  with its rubric text, deliverable scope, and judge explanation when provided.
- Rows contain the actual instructions and requested filenames in `input`,
  and the rubric in `expected`. Setup verifies both against the pinned corpus,
  then stages source documents in bounded ZIP archives through `CaseFile`.
  The archives preserve paths and avoid Beaker's 128-files-per-case limit for
  diligence data rooms with thousands of files. The candidate receives
  no `task.json` or rubric. It uses the staged documents without fetching data.
- Predictions are the extracted text of submitted deliverables. Filename
  matching and partial submissions follow the existing scorer. Judge batches
  contain at most four criteria. Every response must contain exactly one
  pass/fail verdict for each requested criterion; incomplete, duplicate,
  unknown, or invalid verdicts trigger retries. Empty rubrics are rejected
  when preparing a dataset. Exhausted judge failures raise an error instead
  of assigning a misleading zero.

## Dataset and structural validation

Validate two real tasks without uploading a dataset or calling a model:

```bash
uv run python .beaker/upload_splits.py --smoke-only --train-limit 1 --test-limit 1
```

The helper reads the frozen train/test lists and downloads the selected tasks
at `HARVEY_LABS_COMMIT`. It writes JSONL in a temporary directory, runs
`beaker run smoke --strict`, and removes the JSONL when finished. Public task
downloads use the recipe's normal cache; `GITHUB_TOKEN` is optional if GitHub
rate limits become a problem. `--tasks-root /path/to/tasks` reuses a checkout
at the pinned commit.

To upload the default 8 train / 4 test tasks to the selected Beaker target:

```bash
uv run python .beaker/upload_splits.py
```

These prefixes exclude the diligence data rooms with thousands of documents.

Use `--train-limit` and `--test-limit` for other prefix sizes, or `--full`
for all 1660 train / 100 test tasks. Full preparation downloads the complete
corpus and can take considerable time and disk space. Use a distinct `--name`
when changing the dataset size. `--agent` overrides the upload target.

After upload, the helper prints an immutable `name@revision` and validates
that hosted snapshot. Keep that selector for the later run. No dataset selector
or local filesystem path is committed to the YAML.

## Models, scoring, and hosted execution

Without a selected rollout model, Beaker preserves the recipe's default
OpenRouter client and model. Local runs use the application's provider key;
supported hosted calls can use Beaker provider routing. When a run selects a
model, the integration injects Stirrup's native `LiteLLMClient` with
`inference_target(runtime)`. The gateway controls reasoning effort.
The recipe's token and context-window settings remain in effect.

The rubric judge uses `scoring_inference_target()` in hosted runs so its
model and usage are separate from candidate execution. Hosted runs must
explicitly configure `scorer_model`; there is no silent provider fallback.
Local scoring retains the recipe's DeepSeek V4 Flash judge and provider key.
For example, after choosing that same judge for a hosted run:

```bash
uv run beaker run trigger --integration-id harvey-lab \
  --agent harvey-lab-agent --dataset 'DATASET_NAME@REVISION' \
  --ref YOUR_PUSHED_BRANCH \
  --config '{"scorer_model":"openrouter:deepseek/deepseek-v4-flash"}'
```

Run this only when you intend to start a hosted optimization run. Replace the
dataset and branch placeholders with the validated revision and pushed branch.
Keep the scorer model fixed across baseline and candidate evaluations.

The hosted image installs the recipe, the Beaker tracing dependency, `pandoc`,
`poppler-utils`, and LibreOffice. Beaker stays a development dependency for
local use; the normal `harvey-lab` commands do not import it.

LiteLLM tracing is active only during the candidate agent call, including its
turns and context compaction. Tool results appear in subsequent model messages;
separate Stirrup tool execution spans are not added. Judge traffic is excluded
from candidate tracing.

Smoke validates configuration, typed rows, staged files, and setup cleanup. It
does not execute the agent or judge, validate credentials, or establish a score.
As in the standalone recipe, Stirrup's local shell runs as the current user;
hosted candidate execution relies on Beaker's sandbox.
