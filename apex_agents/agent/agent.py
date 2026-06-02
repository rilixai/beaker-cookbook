"""A faithful ReAct toolbelt agent for APEX-Agents.

We are NOT wrapping Mercor's Archipelago harness (a non-pip-installable
distributed service). Instead this is our own faithful ReAct toolbelt
agent, seeded VERBATIM from Archipelago's reference prompts so behavior
matches. The loop structure mirrors Archipelago's reference:

* Messages start ``[{system: rendered system_prompt},
  {user: rendered task_template}]``.
* Tools exposed each turn = META_TOOLS + the current toolbelt +
  ``final_answer``. The agent starts with an EMPTY toolbelt and must
  ``toolbelt_add_tool`` domain tools (matches Archipelago).
* META_TOOLS: ``toolbelt_list_tools``, ``toolbelt_inspect_tool``,
  ``toolbelt_add_tool``, ``toolbelt_remove_tool``, ``todo_write``.
* DOMAIN_TOOLS (over the :class:`WorldFiles` surface): ``list_files``,
  ``read_file``, ``read_spreadsheet``, ``read_pdf``, ``search_files``.
* ``final_answer(answer, status)`` terminates; rejected if open todos
  (faithful to Archipelago).
* ReSum: when estimated message tokens exceed ``TRIGGER_FRACTION`` of
  ``max_context_tokens``, the LLM is called with
  ``resum_summary_prompt.format(conversation=...)``; old messages are
  replaced with the summary and the last ``KEEP_RECENT_MESSAGES`` are
  kept verbatim (faithful to Archipelago resum.py constants).

Prompt strings are constructor configuration for the agent instance. Callers
that want to run a different prompt set construct another agent with those
strings. The model factory is injectable so tests pass a scripted
deterministic model and no real API call fires.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from .types import AgentToolCall, ApexAgentsAgentOutput


logger = logging.getLogger(__name__)


__all__ = [
    "ApexReActAgent",
    "ModelFactory",
    "WorldFactory",
    "build_litellm_model_factory",
]


# ReSum constants — faithful to Archipelago resum.py.
TRIGGER_FRACTION = 0.70
KEEP_RECENT_MESSAGES = 10

# Per-LLM-call timeout + bounded retries. Without these a hung provider
# request blocks the whole run indefinitely (observed: a ~5h stall +
# a timeout-zeroed case). litellm supports `timeout`/`num_retries`
# natively, so a hung call raises after the budget — the agent loop's
# existing `except` then fails that case fast instead of wedging.
_DEFAULT_LLM_TIMEOUT_S = 120.0
_DEFAULT_LLM_NUM_RETRIES = 2

# A todo counts as "closed" (no longer blocking final_answer) if its
# status is any of these. Models naturally emit "done" / "complete" /
# "finished" — restricting the gate to {"completed","cancelled"} caused
# a livelock where the agent marked everything "done", final_answer was
# rejected forever, and it burned its whole step budget thrashing.
_CLOSED_TODO_STATUSES = frozenset(
    {
        "completed",
        "complete",
        "done",
        "finished",
        "closed",
        "resolved",
        "cancelled",
        "canceled",
        "skipped",
        "skip",
        "wontfix",
        "n/a",
        "na",
        "not applicable",
    }
)

# Hard ceiling on consecutive rejected final_answer attempts. Past this
# the gate auto-cancels the remaining todos and accepts the answer, so
# a willing agent can always terminate with its result instead of
# livelocking against the todo gate (a harness defect, not a faithful
# Archipelago behavior).
_MAX_FINAL_ANSWER_REJECTIONS = 3


def _is_todo_closed(status: Any) -> bool:
    return str(status or "").strip().lower() in _CLOSED_TODO_STATUSES


# Per-case world factory: ``(record) -> WorldFiles``-like. Tests inject
# a closure yielding a :class:`FakeWorld`.
WorldFactory = Callable[[Any], Any]
# Per-case model factory: ``(model_name, temperature) -> ChatModel``.
# A ChatModel is any object with ``complete(messages, tools) -> dict``
# returning ``{"content": str, "tool_calls": [...], "cost": float}``.
ModelFactory = Callable[[str, float], Any]


def _render_task_template(task_template: str, task: str) -> str:
    """Substitute the Jinja2 ``{{task}}`` variable.

    A tiny literal substitution (no Jinja2 dependency needed for the
    single ``{{task}}`` variable) — mirrors SWE-bench's instance
    template substitution. If the rewritten template dropped
    ``{{task}}`` the raw task is appended so the agent still sees it.
    """
    if "{{task}}" in task_template:
        return task_template.replace("{{task}}", task)
    if "{{ task }}" in task_template:
        return task_template.replace("{{ task }}", task)
    return f"{task_template}\n\n{task}"


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate (~4 chars/token) over message text.

    Cheap + dependency-free; only needs to be monotone so the ReSum
    trigger fires deterministically in tests.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        total_chars += len(json.dumps(msg.get("tool_calls", []), default=str))
    return total_chars // 4 + 1


# ─── Tool schemas ─────────────────────────────────────────────────────


def _meta_tool_schemas() -> list[dict[str, Any]]:
    """The always-available meta-tools (OpenAI function-calling schema)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "todo_write",
                "description": "Create or update the task todo list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                            },
                        },
                        "merge": {"type": "boolean"},
                    },
                    "required": ["todos"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "toolbelt_list_tools",
                "description": "List the domain tools available to add to the toolbelt.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "toolbelt_inspect_tool",
                "description": "Inspect a domain tool's signature/description.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "toolbelt_add_tool",
                "description": "Add a domain tool to the active toolbelt.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "toolbelt_remove_tool",
                "description": "Remove a domain tool from the active toolbelt.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Submit the final answer. Rejected if todos are incomplete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["completed", "blocked", "failed"],
                        },
                    },
                    "required": ["answer"],
                },
            },
        },
    ]


_DOMAIN_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "list_files": {
        "description": "List files in the world (optionally under a subdir).",
        "parameters": {
            "type": "object",
            "properties": {"subdir": {"type": "string"}},
        },
    },
    "read_file": {
        "description": "Read a UTF-8 text file from the world.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "read_spreadsheet": {
        "description": (
            "Read an .xlsx workbook, one sheet at a time. The reply starts with a "
            "'# Sheets:' index of every tab; the computed model is usually on a LATER "
            "tab, not the first. Pass `sheet` (exact name from the index) to read a "
            "specific tab — omitting it returns only the index + the first sheet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "sheet": {"type": "string", "description": "Exact sheet/tab name to read."},
            },
            "required": ["path"],
        },
    },
    "read_pdf": {
        "description": "Extract text from a PDF in the world.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "read_docx": {
        "description": "Read a .docx (Microsoft Word) document from the world — contracts, surveys, memos. Use this for any .docx path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "search_files": {
        "description": "Case-insensitive substring search over the world's text files.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _domain_tool_schema(name: str) -> dict[str, Any]:
    spec = _DOMAIN_TOOL_SPECS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["parameters"],
        },
    }


class ApexReActAgent:
    """Faithful async ReAct toolbelt agent for one APEX-Agents task.

    The wrapper does NOT keep long-lived loop state — every case needs its own
    world + message history.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_temperature: float = 0.0,
        max_steps: int = 60,
        cost_limit: float = 3.0,
        max_toolbelt_size: int = 80,
        max_context_tokens: int = 120_000,
        default_system_prompt: str,
        default_task_template: str,
        default_resum_summary_prompt: str,
        world_factory: WorldFactory,
        model_factory: ModelFactory | None = None,
        llm_timeout: float = _DEFAULT_LLM_TIMEOUT_S,
    ) -> None:
        self._model_name = model_name
        self._model_temperature = model_temperature
        self._max_steps = max_steps
        self._cost_limit = cost_limit
        self._max_toolbelt_size = max_toolbelt_size
        self._max_context_tokens = max_context_tokens
        self._system_prompt = default_system_prompt
        self._task_template = default_task_template
        self._resum_summary_prompt = default_resum_summary_prompt
        self._world_factory = world_factory
        self._model_factory: ModelFactory = model_factory or build_litellm_model_factory(timeout=llm_timeout)

    def _snapshot_prompts(self) -> tuple[str, str, str]:
        return (
            self._system_prompt,
            self._task_template,
            self._resum_summary_prompt,
        )

    # ─── main entrypoint ──────────────────────────────────────────────

    async def forward(self, *, record: Any) -> ApexAgentsAgentOutput:
        """Run one APEX-Agents task through the ReAct loop end-to-end.

        The world factory may block (zip extraction / HF download), so
        it runs in :func:`asyncio.to_thread`; the sync ReAct loop is
        likewise offloaded so rilixai's async runtime can drive many
        cases concurrently without blocking the event loop.
        """
        started = time.monotonic()
        world: Any = None
        try:
            # Snapshot constructor prompts before the first await so this case
            # uses one internally consistent prompt set.
            sys_p, task_t, resum_p = self._snapshot_prompts()
            world = await asyncio.to_thread(self._world_factory, record)
            output = await asyncio.to_thread(
                self._run_loop,
                record=record,
                world=world,
                system_prompt=sys_p,
                task_template=task_t,
                resum_summary_prompt=resum_p,
            )
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            logger.exception("APEX-Agents agent failed for task %s", getattr(record, "task_id", "?"))
            output = ApexAgentsAgentOutput(
                final_answer="",
                status=type(exc).__name__,
                extra={"error": str(exc)},
            )
        finally:
            if world is not None:
                close = getattr(world, "close", None)
                if callable(close):
                    try:
                        await asyncio.to_thread(close)
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("world.close() raised", exc_info=True)
        output.wall_seconds = time.monotonic() - started
        return output

    # ─── ReAct loop ───────────────────────────────────────────────────

    def _run_loop(
        self,
        *,
        record: Any,
        world: Any,
        system_prompt: str,
        task_template: str,
        resum_summary_prompt: str,
    ) -> ApexAgentsAgentOutput:
        model = self._model_factory(self._model_name, self._model_temperature)
        task_prompt = str(getattr(record, "prompt", "") or "")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _render_task_template(task_template, task_prompt)},
        ]
        toolbelt: list[str] = []
        todos: list[dict[str, Any]] = []
        total_cost = 0.0
        resum_count = 0
        final_answer = ""
        status = "incomplete"
        final_answer_rejections = 0
        extra: dict[str, Any] = {}

        meta_schemas = _meta_tool_schemas()
        step = 0
        while step < self._max_steps:
            step += 1

            # ReSum: compact the conversation if it grew past the
            # trigger fraction of the context budget.
            if _estimate_tokens(messages) > int(TRIGGER_FRACTION * self._max_context_tokens):
                messages, summary_cost = self._resummarize(
                    model=model,
                    messages=messages,
                    resum_summary_prompt=resum_summary_prompt,
                )
                total_cost += summary_cost
                resum_count += 1

            tools = meta_schemas + [_domain_tool_schema(n) for n in toolbelt]
            try:
                response = model.complete(messages=messages, tools=tools)
            except Exception as exc:
                status = type(exc).__name__
                extra["error"] = str(exc)
                break

            content = str(response.get("content") or "")
            tool_calls = list(response.get("tool_calls") or [])
            total_cost += float(response.get("cost") or 0.0)
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if not tool_calls:
                # No action taken — nudge the model to act or finish.
                messages.append(
                    {
                        "role": "user",
                        "content": "No tool call was made. Use a tool or call final_answer.",
                    }
                )
                if self._cost_limit and total_cost >= self._cost_limit:
                    status = "cost_limit"
                    break
                continue

            terminated = False
            for call in tool_calls:
                name = str(call.get("name") or "")
                args = call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, Mapping):
                    args = {}
                call_id = str(call.get("id") or f"call_{step}")

                if name == "final_answer":
                    open_todos = [t for t in todos if not _is_todo_closed(t.get("status"))]
                    if open_todos and final_answer_rejections < _MAX_FINAL_ANSWER_REJECTIONS:
                        final_answer_rejections += 1
                        result_text = (
                            "final_answer rejected: these todos are still open: "
                            f"{[t.get('id') for t in open_todos]}. Mark each one's status "
                            "'completed' (or 'cancelled') via todo_write — re-send the FULL "
                            "todo list with merge=false so the update is unambiguous — then "
                            f"call final_answer again. (Attempt {final_answer_rejections}/"
                            f"{_MAX_FINAL_ANSWER_REJECTIONS}; after that the answer is accepted "
                            "regardless.)"
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": call_id, "name": name, "content": result_text}
                        )
                        continue
                    if open_todos:
                        # Livelock guard: the agent is willing to finish
                        # but cannot satisfy the todo gate. Accept the
                        # answer rather than burn the step budget.
                        logger.warning(
                            "final_answer accepted with %d open todos after %d rejections (livelock guard)",
                            len(open_todos),
                            final_answer_rejections,
                        )
                        extra["forced_final_answer"] = True
                    final_answer = str(args.get("answer") or "")
                    status = str(args.get("status") or "completed")
                    terminated = True
                    break

                result_text = self._dispatch_tool(
                    name=name,
                    args=args,
                    world=world,
                    toolbelt=toolbelt,
                    todos=todos,
                )
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": result_text})

            if terminated:
                break
            if self._cost_limit and total_cost >= self._cost_limit:
                status = "cost_limit"
                break
        else:
            status = "max_steps"

        harvested = _harvest_messages(messages)
        return ApexAgentsAgentOutput(
            final_answer=final_answer,
            status=status,
            messages=harvested,
            total_steps=step,
            total_cost=total_cost,
            resum_count=resum_count,
            extra=extra,
        )

    # ─── tool dispatch ────────────────────────────────────────────────

    def _dispatch_tool(
        self,
        *,
        name: str,
        args: Mapping[str, Any],
        world: Any,
        toolbelt: list[str],
        todos: list[dict[str, Any]],
    ) -> str:
        """Execute one tool call and return its textual result."""
        try:
            if name == "todo_write":
                return self._tool_todo_write(args=args, todos=todos)
            if name == "toolbelt_list_tools":
                return "Available domain tools: " + ", ".join(sorted(_DOMAIN_TOOL_SPECS))
            if name == "toolbelt_inspect_tool":
                target = str(args.get("name") or "")
                spec = _DOMAIN_TOOL_SPECS.get(target)
                if spec is None:
                    return f"Unknown tool {target!r}."
                return f"{target}: {spec['description']} params={json.dumps(spec['parameters'])}"
            if name == "toolbelt_add_tool":
                target = str(args.get("name") or "")
                if target not in _DOMAIN_TOOL_SPECS:
                    return f"Cannot add unknown tool {target!r}."
                if len(toolbelt) >= self._max_toolbelt_size:
                    return "Toolbelt is full; remove a tool before adding another."
                if target not in toolbelt:
                    toolbelt.append(target)
                return f"Added {target!r} to the toolbelt. Active: {toolbelt}."
            if name == "toolbelt_remove_tool":
                target = str(args.get("name") or "")
                if target in toolbelt:
                    toolbelt.remove(target)
                    return f"Removed {target!r}. Active: {toolbelt}."
                return f"{target!r} was not in the toolbelt."
            if name in _DOMAIN_TOOL_SPECS:
                if name not in toolbelt:
                    return f"Tool {name!r} is not in the active toolbelt. Add it with toolbelt_add_tool first."
                return self._dispatch_domain_tool(name=name, args=args, world=world)
            return f"Unknown tool {name!r}."
        except FileNotFoundError as exc:
            return f"File not found: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            return f"Tool {name!r} raised {type(exc).__name__}: {exc}"

    @staticmethod
    def _tool_todo_write(*, args: Mapping[str, Any], todos: list[dict[str, Any]]) -> str:
        incoming = args.get("todos")
        merge = bool(args.get("merge", False))
        if not isinstance(incoming, (list, tuple)):
            return "todo_write requires a `todos` array."
        normalized = [
            {
                "id": str(t.get("id") or ""),
                "content": str(t.get("content") or ""),
                "status": str(t.get("status") or "pending"),
            }
            for t in incoming
            if isinstance(t, Mapping)
        ]
        if merge:
            # Match incoming updates to existing todos by id when the id
            # is non-empty, else by content. Keying purely on id (the
            # old behavior) collapsed every empty-id todo into ONE entry
            # — so the agent could never address individual items to
            # close them, which (with the strict status gate) produced
            # the final_answer livelock. Order is preserved; unmatched
            # incoming todos are appended.
            for t in normalized:
                match = None
                for existing in todos:
                    if t["id"] and existing.get("id") == t["id"]:
                        match = existing
                        break
                    if not t["id"] and existing.get("content") == t["content"]:
                        match = existing
                        break
                if match is not None:
                    match.update(t)
                else:
                    todos.append(t)
        else:
            todos[:] = normalized
        return f"Todos updated ({len(todos)} total): {todos}"

    @staticmethod
    def _dispatch_domain_tool(*, name: str, args: Mapping[str, Any], world: Any) -> str:
        if name == "list_files":
            files = world.list_files(str(args.get("subdir") or ""))
            return "Files:\n" + "\n".join(files) if files else "(no files)"
        if name == "read_file":
            return str(world.read_text(str(args.get("path") or "")))
        if name == "read_spreadsheet":
            sheet_arg = args.get("sheet")
            sheet = str(sheet_arg) if sheet_arg not in (None, "") else None
            return str(world.read_spreadsheet(str(args.get("path") or ""), sheet=sheet))
        if name == "read_pdf":
            return str(world.read_pdf(str(args.get("path") or "")))
        if name == "read_docx":
            return str(world.read_docx(str(args.get("path") or "")))
        if name == "search_files":
            hits = world.search(str(args.get("query") or ""))
            if not hits:
                return "(no matches)"
            return "\n".join(f"{h['file']}:{h['line']}: {h['text']}" for h in hits)
        return f"Unknown domain tool {name!r}."

    # ─── ReSum ────────────────────────────────────────────────────────

    def _resummarize(
        self,
        *,
        model: Any,
        messages: list[dict[str, Any]],
        resum_summary_prompt: str,
    ) -> tuple[list[dict[str, Any]], float]:
        """Compact ``messages`` into a summary, keeping the last N verbatim.

        Faithful to Archipelago resum.py: the system message stays, the
        middle is summarized via ``resum_summary_prompt.format(
        conversation=...)``, and the last ``KEEP_RECENT_MESSAGES``
        non-system messages are kept verbatim.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        if len(non_system) <= KEEP_RECENT_MESSAGES:
            return messages, 0.0
        to_summarize = non_system[:-KEEP_RECENT_MESSAGES]
        recent = non_system[-KEEP_RECENT_MESSAGES:]

        conversation = "\n\n".join(f"[{m.get('role')}] {m.get('content') or ''}" for m in to_summarize)
        prompt = resum_summary_prompt.format(conversation=conversation)
        try:
            response = model.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
            )
            summary_text = str(response.get("content") or "")
            summary_cost = float(response.get("cost") or 0.0)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ReSum summarization failed: %s", exc)
            return messages, 0.0

        compacted: list[dict[str, Any]] = list(system_msgs)
        compacted.append(
            {
                "role": "user",
                "content": f"[Compacted reasoning state from earlier in the session]\n{summary_text}",
            }
        )
        compacted.extend(recent)
        return compacted, summary_cost


# ─── message harvesting ───────────────────────────────────────────────


def _harvest_messages(raw_messages: list[dict[str, Any]]) -> list[AgentToolCall]:
    """Convert the internal message list into ``AgentToolCall`` rows."""
    out: list[AgentToolCall] = []
    for idx, msg in enumerate(raw_messages or []):
        role = str(msg.get("role") or "")
        content = msg.get("content")
        content_str = content if isinstance(content, str) else json.dumps(content, default=str)
        tool_name: str | None = None
        tool_args: dict[str, Any] | None = None
        output: str | None = None
        if role == "assistant":
            calls = msg.get("tool_calls") or []
            if calls:
                first = calls[0]
                tool_name = str(first.get("name") or "") or None
                raw_args = first.get("arguments")
                if isinstance(raw_args, Mapping):
                    tool_args = dict(raw_args)
                elif isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                        tool_args = parsed if isinstance(parsed, dict) else None
                    except Exception:
                        tool_args = None
        if role == "tool":
            tool_name = str(msg.get("name") or "") or None
            output = content_str
        out.append(
            AgentToolCall(
                step_index=idx,
                role=role,
                content=content_str,
                tool_name=tool_name,
                tool_args=tool_args,
                output=output,
            )
        )
    return out


# ─── default litellm-backed model factory ─────────────────────────────


def _to_openai_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape internal history into the OpenAI chat-completions schema.

    The loop stores assistant tool calls in a flat internal shape
    (``{"id", "name", "arguments"}``) for convenience. The OpenAI
    (and litellm-proxied) chat API requires the *nested* shape on
    assistant turns::

        {"id": ..., "type": "function",
         "function": {"name": ..., "arguments": "<json string>"}}

    Replaying the flat shape verbatim makes the API reject the second
    request with ``BadRequestError`` — which previously killed every
    multi-step task right after its first tool round-trip. We also:

    * set assistant ``content`` to ``None`` when it carries tool calls
      (OpenAI wants null, not ``""``), and
    * JSON-encode ``arguments`` if a dict slipped through, and
    * drop empty tool_calls keys so plain turns stay clean.

    Pure function (no network) so it is unit-testable on its own.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        m2 = {k: v for k, v in m.items() if k != "tool_calls"}
        raw_calls = m.get("tool_calls") or []
        if raw_calls:
            nested: list[dict[str, Any]] = []
            for tc in raw_calls:
                args = tc.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, default=str)
                nested.append(
                    {
                        "id": tc.get("id") or "",
                        "type": "function",
                        "function": {"name": tc.get("name", ""), "arguments": args},
                    }
                )
            m2["tool_calls"] = nested
            # OpenAI requires null (not "") content alongside tool_calls.
            if not m2.get("content"):
                m2["content"] = None
        out.append(m2)
    return out


def build_litellm_model_factory(
    *,
    timeout: float = _DEFAULT_LLM_TIMEOUT_S,
    num_retries: int = _DEFAULT_LLM_NUM_RETRIES,
) -> ModelFactory:
    """Return a factory building a thin litellm-backed chat model.

    The returned ChatModel exposes ``complete(messages, tools) ->
    {"content": str, "tool_calls": list, "cost": float}``. litellm is
    imported lazily inside :meth:`complete` so this module imports
    offline; tests inject their own scripted model and never reach
    this path.
    """

    def _factory(model_name: str, temperature: float) -> Any:
        return _LitellmChatModel(
            model_name=model_name,
            temperature=temperature,
            timeout=timeout,
            num_retries=num_retries,
        )

    return _factory


class _LitellmChatModel:
    """Minimal litellm function-calling wrapper used in production runs."""

    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        timeout: float = _DEFAULT_LLM_TIMEOUT_S,
        num_retries: int = _DEFAULT_LLM_NUM_RETRIES,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature
        self._timeout = timeout
        self._num_retries = num_retries

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import litellm  # lazy import — keeps the module offline-importable

        api_messages = _to_openai_api_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": api_messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
        kwargs["timeout"] = self._timeout
        kwargs["num_retries"] = self._num_retries
        response = litellm.completion(**kwargs)
        choice = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(choice, "tool_calls", None) or []:
            tool_calls.append(
                {
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc.function, "name", ""),
                    "arguments": getattr(tc.function, "arguments", "{}"),
                }
            )
        cost = 0.0
        try:
            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # pragma: no cover - cost is best-effort
            cost = 0.0
        return {
            "content": getattr(choice, "content", "") or "",
            "tool_calls": tool_calls,
            "cost": cost,
        }
