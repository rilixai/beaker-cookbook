# Agents' Last Exam

Beaker integration for all **122 public ALE v1.1 tasks** selected by upstream
`selected_tasks/full.txt`, pinned to `2e91edd3cd26baa35a38127c90c8e5241dfbe62f`.

## Dataset review

Rows retain the public task prompt, grading description, required outputs,
reference-file descriptions, and public evaluator source and its file hashes.
`manifest.json` declares each task's pinned Git directory so the reviewer can
read supported documents on demand. Python, shell, TOML, and YAML source is
included in the rows because the current lazy reader does not support those extensions. No benchmark source is vendored.

All tasks remain in `test`; `train` is empty. ALE does not provide a canonical
optimization training split. This upload does not create one.

Public input archives live on Hugging Face. Reference archives require upstream
access approval. Neither is included in the upload or available through the
reviewer's Git resource reader. The rows preserve both pinned upstream locations
and this evidence limitation. Findings requiring those documents remain unverified.

```bash
uv sync --group dev --locked
# Use a clean upstream checkout at the commit above.
uv run python .beaker/export_dataset.py --source /tmp/ale-source --destination /tmp/ale-dataset
uv run beaker run smoke --strict --split test --config '{"local_dataset_path":"/tmp/ale-dataset"}'
uv run beaker dataset upload /tmp/ale-dataset --name agents-last-exam-v1.1-full \
  --agent agents-last-exam --split train=0 --split test=122 --total-count 122
```

Set `BEAKER_API_BASE_URL` to the intended preview before hosted commands.
Use the immutable dataset revision returned by upload for later validation.

## Rollouts

The adapter delegates execution and scoring to the **unmodified upstream ALE
runner**, one subprocess and output directory per case. The editable target is
`src/agents_last_exam/agent.yaml`; graders and environment settings are fixed.
Infrastructure and evaluator failures raise errors instead of becoming score zero.

Rollouts require a fully provisioned ALE installation at `ALE_CHECKOUT_PATH`
matching the pinned commit, and `ALE_ENVIRONMENT_CONFIG` pointing to its environment
YAML. Follow upstream setup for VM images, task archives, reference access, native
dependencies, provider credentials, and any judge credentials. These resources are
not provisioned by this recipe. The preset retains upstream Claude Code/OpenRouter
settings; Beaker model comparison and model-level tracing are not wired.
Structural smoke does not execute the runner or prove VM readiness.

[Upstream code and setup](https://github.com/rdi-berkeley/agents-last-exam)
Software: Apache-2.0. Benchmark data: CC-BY-4.0.
