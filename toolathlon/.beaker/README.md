# Beaker execution

Beaker may edit **`prompts/agent_system_prompt.md` only**. Its seed is the original
system prompt shared by the three quickstart tasks. Bootstrap, dependency pins,
tool implementations and graders remain outside that editable surface.

Both onboarding and the standalone CLI execute the original
`main.py → TaskRunner → TaskAgent`, migrated in place to **OpenAI Agents 0.20.0**.
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

`toolathlon_entry.py` surrounds upstream `main()` with native trace export.
`--agent_only` skips inline evaluation so the trusted parent can run the official
grader after the sandbox exits. Standalone CLI runs keep inline evaluation.
Task instructions, initial files, tools and evaluator logic stay upstream.

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

Onboarding requires Linux x86_64/aarch64 with **Landlock ABI >= 3**. The policy allows
system runtimes and dependencies read-only, plus per-case writable directories.
Staged tasks contain instructions, initial files and preprocessing; local reference
files and other cases are inaccessible. The unmodified Excel package is copied per
case because it writes a log beside itself. Network access remains available,
including public benchmark data. Unsupported isolation fails explicitly.

Official scoring uses trusted reference hashes and the complete returned workspace.
Oversized workspaces fail explicitly instead of dropping files. Failures before a
model turn are retryable; exhausted turn budgets are scored as failures even if
upstream leaves its request counter at zero. Cancellation and timeout terminate the
whole case process group, including MCP servers.

## Tasks outside quickstart

The three quickstart tasks keep their native path. Every other official task
uses `toolathlon_docker.py`, selected from its canonical task ID. Dataset metadata
alone cannot enable execution. Docker availability and the task image are checked
during case loading, before baseline model calls.

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

Full task scoring returns the official verdict and bounded diagnostic text instead
of transferring large workspaces through the native 4 MiB context limit. Missing
results, failed bootstrap and evaluator infrastructure errors do not become
measured zero scores. Output records the runner and resolved task-image ID.

The 38-task no-external-account subset still requires local application setup.
The remaining tasks require dedicated third-party accounts. For these, configure
`TOOLATHLON_CONFIG_JSON` in the agent environment as a JSON object mapping paths
relative to upstream `configs/` to their text contents; alternatively a managed
runtime can mount an existing `TOOLATHLON_CONFIG_DIR`. Keep credentials out of Git.
`TOOLATHLON_DOCKER_IMAGE` can select a compatible pinned task-image digest.

Containerized execution retains upstream's privileged Docker/socket requirements;
it does not provide the native Landlock boundary. Use dedicated benchmark accounts
and VM resources. Real Docker execution must be validated before a broad baseline.
