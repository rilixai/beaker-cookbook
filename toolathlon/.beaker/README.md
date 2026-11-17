# Beaker execution

Beaker may edit `prompts/agent_system_prompt.md` and `harness/agent.patch.md`.
The prompt seed is the original system prompt shared by the three quickstart
tasks. The harness patch seed is empty: it preserves the original agent migrated
in place to OpenAI Agents 0.20.0.

To launch from the Beaker platform, select the existing `toolathlon` agent, the
GitHub branch containing this integration, and an existing Toolathlon dataset
revision. Beaker supplies that dataset's rows to this integration. The recipe
validates them against the pinned task definitions; it neither creates splits
nor uploads or selects a dataset. Existing dataset schemas remain unchanged.

The full editable harness surface, source references and patch-generation workflow
are documented in [harness/README.md](../harness/README.md). The candidate patch is
applied only during the agent phase; original code is restored before grading.

Onboarding uses the original prepared-agent phase
`container_agent.run_agent_phase → TaskRunner → TaskAgent`, migrated in place to
**OpenAI Agents 0.20.0**. The standalone CLI uses the same original agent lifecycle.
Preparation applies `src/toolathlon_baseline/upstream/openai-agents-0.20.patch`
and verifies its Git blob hashes. No alternate agent or SDK compatibility layer
is used. The default Beaker cache is keyed by the source revision and migration;
`BEAKER_TOOLATHLON_CACHE` can select an explicitly managed cache.

The baseline provider, MCP registration and context runner are migrated to the
new APIs. Public lifecycle hooks and `RunConfig.call_model_input_filter` preserve
Toolathlon's history, turn accounting, truncation, context-error recovery and turn
budget. Tool aliases, argument coercion, output limits and completion behavior
remain those of the original agent until a candidate changes them.

The upstream `uv.lock` covers two exclusive groups: `agent` (default) runs Agents
0.20.0, MCP 1.19.0, Pydantic 2.12.3 and `beaker-sdk[openai-agents]==0.6.6`; `excel`
runs the unchanged Excel 0.1.4 server with MCP 1.9.0 in `.venv-excel`. Its FastMCP
constructor requires that older MCP. Python stays at 3.12.11. The cookbook parent
installs only Beaker's `tracing` extra; agent dependencies belong to upstream.
This is an SDK-migrated baseline, with updated transitive dependencies.

`toolathlon_container_agent.py` surrounds the original prepared-agent phase with
native trace export. The official phased container runner invokes preparation and
evaluation separately, keeping grader calls outside the agent trace. Task
instructions, initial files, tools and evaluator logic stay upstream.

## Tracing and isolation

The child registers Beaker's native Agents processor before task execution.
Framework spans record actual model/tool timings, inputs/outputs and per-call
usage, including reported cached tokens. The default OpenAI exporter is replaced
by a local, per-case OTLP receiver. The child gets a capture token; only the parent
holds Beaker run credentials.

The parent adopts received batches into its capture, preserving timing and child
hierarchy, adding trusted case/candidate IDs and storing large bodies as artifacts.
Managed captures use Beaker's OTLP ingestion; local captures archive the same
format. Traces are not reconstructed from `traj_log.json` or estimated token splits.
Exporters flush in `finally`, including errors and graceful SIGTERM. Dropped or
rejected batches and missing flush confirmation mark telemetry incomplete. SIGKILL
may prevent a final flush; already received spans remain available. MCP spans
measure calls from the agent; server internals and internal HTTP retries are visible
only where the framework emits spans.

Failures before a model turn are retryable; exhausted turn budgets are scored as
failures even if upstream leaves its request counter at zero. Cancellation and
timeout stop only the current execution’s task containers. Shared applications remain available to other cases.

## Diagnosing a failed case

Each case attaches diagnostic artifacts named `toolathlon.logs.*` to its Beaker
trace, including failures before the first model turn. Open or download the case
trace and its artifacts; `toolathlon.logs.manifest` lists the available files,
their parts and any truncation. These logs are independent of the agent's OTLP
export, so a failed agent bootstrap does not need to emit framework spans.

The collection includes supervisor stdout/stderr, the Python failure traceback,
timed bootstrap/resource-lock steps, Docker command output, the shared daemon
log, the case runner’s `run.log`, official task logs and startup logs for the
applications that case uses. Failure probes record VM interfaces, listeners,
Docker bridge information, effective UID/capabilities and disk space. The
collector never dumps all other cases’ container logs. A short error excerpt
is printed to worker logs as well.

Logs are attached before temporary directories are deleted, on success, failure,
timeout and graceful cancellation. Retention is capped at 8 MiB per file and
32 MiB per case, preserving the beginning and end when truncated; probes have
short time limits. SIGKILL or loss of the entire VM can prevent final collection.
Known environment/configuration secrets and common credential formats are masked
before publication. The collector does not archive environment dumps, credential
files or arbitrary workspace files. Run preparation before a case starts remains
visible in worker logs rather than per-case artifacts.

## Docker execution for every task

Every Beaker task uses the official phased container runner, including
`find-alita-paper`, `excel-market-research` and `interview-report`. Each execution
gets a fresh task container and working copy. Case metadata is normalized to
`runner: containerized`, so existing datasets labeled `native` remain usable.
The standalone CLI's native quickstart remains available separately.

The actual rollout VM owns one Docker daemon, its image cache and its shared
applications. Docker preparation runs lazily in that VM rather than during the
coordinator's dataset loading. There are no per-case Docker hosts, nested daemons,
image archives, daemon-root clones or image imports. A dependency image based on
the official task image installs the pinned migrated agent and the separate Excel
environment once. It is keyed by the resolved base-image ID and dependency files.
Kind, kubectl and Helm host tools are copied from the official image once.

Canvas, email, WooCommerce and Kubernetes are started on first use under a
process-wide setup lock. Later cases reuse the running applications and generated
configuration. A stopped shared service produces an infrastructure error rather
than being restarted underneath active cases. Services live for the rollout VM's
lifetime; destroy the VM to reset the full environment for a new formal campaign.
The original per-task preprocessing still prepares the task's application state.

Resource locks span preprocessing, agent execution, grading and cleanup. They
serialize repeated executions of the same task and upstream's conflict groups.
Kubernetes tasks additionally share one lock because their Kind preparation may
delete clusters or rewrite kubeconfigs. Independent tasks can overlap, with Beaker
still owning case admission (default concurrency 6). Lease records let the next
owner clean an orphaned task container if its supervisor was forcibly killed.

The runner retains upstream's host networking and Docker socket mount. Task
containers reach the Beaker VM's existing provider proxy and per-case OTLP
receiver directly; provider hostname mappings, CA files and capture tokens are
forwarded by the reviewed shell patch. No socat relay or bridge gateway discovery
is needed. Run credentials remain in the parent process.

The original `scripts/run_single_containerized.sh` keeps the same live task
container through preprocessing, reference hiding, the prepared original agent,
trusted restoration and official grading. Candidate harness patches are applied
inside that task container only, then restored before grading. Shared trusted
source, application setup and dependency environments are not candidate-editable.
Cancellation removes only containers matching the case's unique prefix and output
mounts, while keeping the shared daemon, applications and other cases alive.

Scoring returns the official verdict and bounded diagnostic text without
transferring the complete workspace through the case context. Missing
results, failed bootstrap and evaluator infrastructure errors do not become
measured zero scores. Output records the runner and resolved task-image ID.

The 38-task no-external-account subset still requires local application setup.
The remaining tasks require dedicated third-party accounts. For these, configure
`TOOLATHLON_CONFIG_JSON` in the agent environment as a JSON object mapping paths
relative to upstream `configs/` to their text contents; alternatively a managed
runtime can mount an existing `TOOLATHLON_CONFIG_DIR`. Keep credentials out of Git.
`TOOLATHLON_DOCKER_IMAGE` can select a compatible pinned task-image digest.

Containerized execution retains upstream's Docker/socket trust model and shared application state.
Use dedicated benchmark accounts and VM resources. Real Docker execution must be
validated before a broad baseline.
