# Native onboarding execution boundary

Beaker may edit `prompts/agent_system_prompt.md` only. Its initial contents are
the unchanged system prompt shared by the three pinned quickstart tasks.
Preparation, launch adapters, dependencies, reference files and graders stay
outside that editable surface. Expanding the surface to tool implementations
requires a separate, explicit integration change.

The agent runs the original single-turn `TaskRunner`, without evaluation in its
process. Its staged task contains instructions, initial files and preprocessing
only. Scoring runs the official evaluator separately, against trusted reference
hashes and the complete returned workspace. The standalone CLI is unchanged.

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

These checks validate the onboarding boundary; they do not establish parity
with the full 108-task containerized benchmark or the accuracy of reconstructed
per-call traces.
