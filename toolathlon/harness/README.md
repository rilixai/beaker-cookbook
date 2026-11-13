# Editable Toolathlon agent harness

The patch can modify the original agent loop, TaskRunner, lifecycle hooks, context
management, model providers/retries, local tools, MCP registration/configuration,
logging and supporting agent utilities. It may also add or modify local MCP server
source under `local_servers/`. The exact file/directory allowlist lives in
`../.beaker/toolathlon_harness.py`; task definitions, references, preprocessors, graders,
Docker bootstrap, tracing plumbing and dependency locks are excluded.

Beaker edits a standard Git diff against the **SDK-migrated** upstream source,
which is available in the prepared cache. This keeps the complete solver code
editable without copying thousands of upstream source lines into the recipe.
The patch document lives outside `.beaker/`, which Beaker protects from candidate
edits. Its Markdown extension keeps it visible in Beaker’s source exposure; its
contents are a raw Git diff, not a fenced code block. Use a scratch checkout rather
than editing the shared, verified benchmark cache.
For example, from the recipe directory:

```bash
RECIPE_DIR="$PWD"
uv run toolathlon-baseline prepare --cache /tmp/toolathlon-harness-source
git clone --no-hardlinks /tmp/toolathlon-harness-source/upstream /tmp/toolathlon-harness-edit
git -C /tmp/toolathlon-harness-edit apply "$RECIPE_DIR/src/toolathlon_baseline/upstream/openai-agents-0.20.patch"
git -C /tmp/toolathlon-harness-edit add -A
# Edit agent files in the scratch checkout. For new files, use git add -N.
git -C /tmp/toolathlon-harness-edit diff > "$RECIPE_DIR/harness/agent.patch.md"
```

During execution, the official runner first prepares the task and hides its
references using the unmodified benchmark. The trusted host then backs up files
that the candidate patch touches and applies the patch inside the task container.
The tracing entrypoint imports and executes that modified original agent. After
it exits, including an agent failure, the host restores the original files,
removes candidate-added modules and clears their Python bytecode before the
official evaluator runs. A failed patch/restore is an infrastructure error.
The backup and patch staging helper stay outside the task container.

Shared caches and dataset hashes continue to use the pinned benchmark. Candidates
cannot alter the official evaluator through the editable patch. As with the
upstream Docker runner, this is not a security boundary against arbitrary agent
code controlling its privileged task environment.

The editable upstream paths are:

```text
scripts/containerized/container_agent.py
utils/__init__.py
utils/status_manager.py
utils/api_model/                 utils/aux_tools/
utils/conversation/              utils/data_processing/
utils/data_structures/           utils/general/
utils/logging/                   utils/mcp/
utils/misc/                      utils/openai_agents_monkey_patch/
utils/roles/                     utils/system_prompts/
utils/task_runner/               configs/mcp_servers/
local_servers/
```

The baseline is [Toolathlon at the pinned revision](https://github.com/hkust-nlp/Toolathlon/tree/9be8d8fe07a497b18ee61e3f2ae694e9797f39eb),
plus [the versioned SDK migration](https://github.com/rilixai/beaker-cookbook/blob/86f9f9f3eb1beff87a2d03246294004b45ad8b27/toolathlon/src/toolathlon_baseline/upstream/openai-agents-0.20.patch).
Dependency manifests and locks inside these directories remain excluded. The
patch document is the optimizer's editable artifact; this README is visible
source context and remains outside the optimization target.
