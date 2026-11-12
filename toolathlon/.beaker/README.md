# Native onboarding execution boundary

Beaker may edit `prompts/agent_system_prompt.md` only. Its initial contents are
the unchanged system prompt shared by the three pinned quickstart tasks.
Preparation, launch adapters, dependencies, reference files and graders stay
outside that editable surface. Expanding the surface to tool implementations
requires a separate, explicit integration change.

The onboarding agent uses **OpenAI Agents 0.20.0** and Beaker's native Agents
connector (`beaker-sdk[openai-agents]==0.6.6`). Its separate environment is locked
in `agent_runtime/uv.lock`; MCP 1.19.0 and Pydantic 2.12.3 preserve compatibility
with the benchmark's older MCP servers. The agent has its own locked dependencies;
it does not inherit the original agent's Python packages. MCP executables and
Python data-processing tools still use the benchmark's original environment.
The recipe pins Python 3.12.11 for local development and CI as well; newer Python
releases are not assumed compatible with these historical dependency pins.
The official graders and the standalone baseline CLI retain that original
environment. This is an agent-framework migration, so onboarding results are a
modernized variant, not an exact reproduction of the original 0.0.15 baseline.

The onboarding entrypoint directly builds an `Agent` with
`OpenAIChatCompletionsModel` and calls `Runner.run`. It does not execute or patch
the legacy `TaskRunner`, `TaskAgent`, model provider or SDK internals.
`toolathlon_context.py` implements history, turn accounting and context truncation
with public `RunHooks` and `RunConfig.call_model_input_filter`. Task completion
via `claim_done` uses the SDK's `stop_at_tool_names` behavior. Context-length
errors reset the model context with the task and recent history, within the
remaining turn budget.

`toolathlon_tools.py` registers the original local Toolathlon tools and the
real MCP tools as `FunctionTool` instances, with canonical tool names, structured
argument coercion and bounded outputs saved for the overlong-output tools.
Model settings are passed through the SDK's public `ModelSettings`; the normal
SDK model request, retry and tracing behavior applies. There are no generated
source replacements, global SDK method assignments or runtime monkey-patches.

The staged task contains instructions, initial files and preprocessing only.
Scoring runs the official evaluator separately, against trusted reference hashes
and the complete returned workspace. The task assets and tool implementations
are upstream; this native agent is a modernized execution harness for them.

## Native traces

The agent registers Beaker's OpenAI Agents processor before task execution.
Model/tool spans are created by the running framework, with actual timestamps,
per-call usage (including reported cached tokens), and recorded inputs/outputs.
The default OpenAI trace exporter is replaced; only the local, per-case OTLP
receiver receives these traces. The agent receives a capture token, not the
parent's Beaker run credentials.

The parent adopts the OTLP batches into its existing Beaker capture, preserving
the measured times and child hierarchy, stamping trusted case/candidate IDs and
moving large bodies into parent-owned artifacts. The managed capture then uses
Beaker's normal OTLP ingestion route; local captures archive the same format.
There is no replay of `traj_log.json` and no evenly divided token estimate.

The SDK processors and OpenTelemetry exporter are flushed in `finally`, also
on agent errors and graceful SIGTERM. Missing exports, dropped/rejected batches
and missing flush confirmation are reported as incomplete telemetry. A forced
SIGKILL can still prevent the final flush; already received spans are retained.
MCP spans describe calls as observed by the agent, not the internals of each
server. Internal HTTP retries are visible only where the framework emits spans.

The onboarding agent and its MCP children require Linux x86_64 or aarch64 with
Landlock ABI 3 or newer enabled. The filesystem allowlist permits system runtime
files and pinned dependencies read-only, plus per-case writable directories.
It excludes the reference cache and other cases. Policies that would expose
the cache through an allowed parent directory are rejected. The unmodified
Excel MCP package is relocated per case because it writes a log next to itself.
Network access remains available, including access to public benchmark data;
this boundary protects local references, not against downloading public answers.
If Landlock is unavailable, execution fails explicitly rather than falling back
to an agent with unrestricted filesystem access. Verify this requirement on the
actual hosted worker before launching an optimization.

Workspace transport retains a 4 MiB raw-byte limit to fit the base64 payload in
Beaker's context envelope. Exceeding that limit is a technical error, never a
partially transferred workspace scored as a model failure. Symlinks are rejected
so collection cannot dereference private files outside the agent workspace.
Missing scoring context is also a technical error. A task that reaches its model
turn limit is scored as a failure even when upstream leaves its request counter
at zero. Cancellation and supervisor timeout terminate the whole process group,
including MCP servers.
