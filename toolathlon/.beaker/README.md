# Beaker execution

Beaker may edit **`prompts/agent_system_prompt.md` only**. Its seed is the original
system prompt shared by the three quickstart tasks. Bootstrap, dependency pins,
tool implementations and graders remain outside that editable surface.

Onboarding uses the original prepared-agent phase
`container_agent.run_agent_phase → TaskRunner → TaskAgent`, migrated in place to
**OpenAI Agents 0.20.0**. The standalone CLI uses the same original agent lifecycle.
Preparation applies `src/toolathlon_baseline/upstream/openai-agents-0.20.patch`
and verifies its Git blob hashes. No alternate agent or SDK compatibility layer
is used. The default Beaker cache is keyed by the source revision and migration;
`BEAKER_TOOLATHLON_CACHE` can select an explicitly managed cache.

The original provider, MCP registration and context runner are migrated to the
new APIs. Public lifecycle hooks and `RunConfig.call_model_input_filter` preserve
Toolathlon's history, turn accounting, truncation, context-error recovery and turn
budget. Tool aliases, argument coercion, output limits and completion behavior
remain those of the original agent.

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
timeout stop the private benchmark host and its task containers.

## Docker execution for every task

Every Beaker task uses `toolathlon_docker.py`, including `find-alita-paper`,
`excel-market-research` and `interview-report`. There is no native execution or
scoring fallback. Docker availability and the task image are checked during case
loading, before baseline model calls. The standalone CLI's native quickstart
remains available separately.

Case metadata is normalized to `runner: containerized`, including existing dataset
rows labeled `native`. Execution does not depend on that old label, so an existing
quickstart dataset does not need to be uploaded again.

The Beaker image installs Docker and socat and needs a VM sandbox. Each case gets
a private Docker-in-Docker host with its own network, image-store volume and
application state. Only its required Canvas, email, WooCommerce or Kubernetes
services are started, using the upstream setup scripts. The shared VM caches the
public task image; private daemons load it without repeated registry downloads.
Case cancellation removes the host and its anonymous image-store volume.

The original `scripts/run_single_containerized.sh` keeps its phased workflow:
preprocess, hide reference/evaluator files, run the original agent, restore trusted
artifacts, and run the official grader while the same applications remain alive.
A small reviewed shell patch forwards provider host mappings, public CA files and
the case's OTLP token. The tracing entrypoint wraps the original prepared-agent
phase only. No private proxy key or local `.env` is mounted into the Docker host.

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

Containerized execution retains upstream's privileged Docker/socket requirements.
Use dedicated benchmark accounts and VM resources. Real Docker execution must be
validated before a broad baseline.
