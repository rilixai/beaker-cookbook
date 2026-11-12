# Native onboarding execution boundary

Beaker may edit `prompts/agent_system_prompt.md` only. Its initial contents are
the unchanged system prompt shared by the three pinned quickstart tasks.
Preparation, launch adapters, dependencies, reference files and graders stay
outside that editable surface. Expanding the surface to tool implementations
requires a separate, explicit integration change.

Both onboarding and the standalone CLI execute Toolathlon's original
`main.py` → `TaskRunner` → `TaskAgent`. Preparation applies the versioned source
patch `src/toolathlon_baseline/upstream/openai-agents-0.20.patch` once to the
pinned checkout. The original agent is migrated **in place** to **OpenAI Agents
0.20.0**; no alternate agent, runtime SDK shim, or second agent environment is
used. Preparation verifies the exact patch on every native cache reuse.

The patch migrates the original provider, MCP registration and context runner
where SDK APIs changed. The context runner uses public lifecycle hooks and
`RunConfig.call_model_input_filter`, preserving upstream's truncation policies,
turn boundaries, persistent history, context-error reset and remaining-turn
budget. Local and MCP tools retain their original implementations, argument
coercion, aliases and output limits. Completion still follows TaskAgent's
original loop, including its `claim_done` behavior.

The upgraded upstream `.venv` runs the original agent, Python tools and official
graders. The unchanged Excel 0.1.4 MCP server alone retains MCP 1.9.0 in its
own locked server environment (`utils/local_servers/excel_runtime`), because
its old FastMCP constructor is incompatible with MCP 1.19.0. This environment
contains no agent SDK or alternate agent. Other externally installed MCP
servers retain their own environments, as in upstream.

The migrated agent’s upstream `pyproject.toml` and `uv.lock` pin
Agents 0.20.0, Beaker's native connector (`beaker-sdk[openai-agents]==0.6.6`),
MCP 1.19.0 and Pydantic 2.12.3. Python remains pinned to 3.12.11. Required
transitive dependency upgrades are recorded in that lockfile. Benchmark results
therefore identify an SDK-migrated baseline, rather than the historical 0.0.15
software environment.

`toolathlon_entry.py` only surrounds the original entrypoint with native trace
export. The `--agent_only` flag skips inline evaluation for onboarding, where
the trusted parent performs official scoring after the sandbox exits. Normal
CLI execution keeps upstream's inline evaluation.

The staged task contains instructions, initial files and preprocessing only.
Scoring runs the official evaluator separately, against trusted reference hashes
and the complete returned workspace. The task assets, prompts, tool
implementations and evaluator logic remain upstream.

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
