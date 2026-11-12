# Native onboarding execution boundary

Beaker may edit `prompts/agent_system_prompt.md` only. Its initial contents are
the unchanged system prompt shared by the three pinned quickstart tasks.
Preparation, launch adapters, dependencies, reference files and graders stay
outside that editable surface. Expanding the surface to tool implementations
requires a separate, explicit integration change.

The onboarding agent uses **OpenAI Agents 0.20.0** and Beaker's native Agents
connector (`beaker-sdk[openai-agents]==0.6.6`). Its separate environment is locked
in `agent_runtime/uv.lock`; MCP 1.19.0 and Pydantic 2.12.3 preserve compatibility
with the benchmark's older MCP servers. Remaining task libraries come from the
original locked environment, after the upgraded dependencies on `sys.path`.
The official graders and the standalone baseline CLI retain that original
environment. This is an agent-framework migration, so onboarding results are a
modernized variant, not an exact reproduction of the original 0.0.15 baseline.

The agent still uses Toolathlon's single-turn `TaskRunner`, without evaluation
in its process. Its staged task contains instructions, initial files and
preprocessing only. `toolathlon_sdk_compat.py` applies checked replacements to
the staged source: it removes `set_tracing_disabled(True)`, adapts model/MCP
signatures and moves context/history bookkeeping onto the new SDK's execution
path. The original truncation/history helpers and local-tool output limit are
retained. Scoring runs the official evaluator separately, against trusted
reference hashes and the complete returned workspace.

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

## Verification without paid model calls

Run `uv run pytest -q` from the recipe directory for transport and regression
checks. The OTLP tests exercise native SDK spans in a subprocess, asymmetric
token counts, real elapsed times, flush on failure, large bodies and token auth.

For the real Toolathlon loop and MCP checks, prepare a cache once, then run:

```bash
uv run toolathlon-baseline prepare --cache /tmp/toolathlon-native-check
TOOLATHLON_TEST_CACHE=/tmp/toolathlon-native-check uv run pytest tests/test_native_toolathlon.py -q
```

These opt-in checks use a local deterministic model endpoint, exercise the three
quickstart tasks, actual MCP writes, context truncation and the turn limit, and
verify that tools still cannot read local references. They do not measure model
quality or establish parity with the full 108-task containerized benchmark.
