# Terminal-Bench Science

Beaker integration for **96 tasks** on upstream commit
`d4f6c91be7eba71a9fc279c3809317489a8dd374`. This is the pinned main-branch
collection, not the older 70-task v0.1 release.

## Dataset review

Rows preserve task instructions, task configuration, full-credit reward, and a
public verifier file inventory. The dataset manifest declares each pinned task
directory. The reviewer can read tests, environment files, and other public
evidence on demand. Large scientific assets are not copied into the upload.
External downloads and unsupported binary formats can still leave evidence gaps.

All 96 tasks are in `test`; `train` is empty. No optimization training split is
invented. Invalid upstream configuration is preserved with its parse error so a
review can diagnose the task rather than silently dropping it.

```bash
uv sync --group dev --locked
# Use a clean upstream checkout at the commit above.
uv run python .beaker/export_dataset.py --source /tmp/science-source --destination /tmp/science-dataset
uv run beaker run smoke --strict --split test --config '{"local_dataset_path":"/tmp/science-dataset"}'
uv run beaker dataset upload /tmp/science-dataset --name terminal-bench-science-full \
  --agent terminal-bench-science --split train=0 --split test=96 --total-count 96
```

Set `BEAKER_API_BASE_URL` to the intended preview before hosted commands.
Use the immutable dataset revision returned by upload for later validation.

## Rollouts

The adapter uses Harbor and the cookbook Terminus agent. It downloads only the
selected task through a filtered Git cache, stages the upstream environment and
tests without oracle solutions, and returns the upstream reward in [0, 1].
Only `src/terminal_bench_science` is editable during optimization.

Execution needs Docker/Compose and enough resources for the selected scientific
task. Some tasks need GPUs, large downloads, or long runtimes. No full benchmark
rollout has been run as part of this review setup. Structural smoke validates rows
and the integration contract; it does not certify those environments or graders.

[Upstream tasks](https://github.com/harbor-framework/terminal-bench-science)
License: Apache-2.0. The Terminus prompt files retain their upstream notices.
