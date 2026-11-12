"""Toolathlon tools registered directly as public Agents SDK FunctionTools."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from agents import FunctionTool
from agents.tracing import get_current_span
from utils.aux_tools.basic import tool_done
from utils.aux_tools.context_management_tools import context_management_tools
from utils.aux_tools.history_tools import history_tools
from utils.aux_tools.overlong_tool_manager import overlong_tool_tools
from utils.aux_tools.python_interpretor import tool_python_execute
from utils.openai_agents_monkey_patch.tool_name_aliases import to_model_mcp_tool_name, to_model_tool_name


LOCAL_TOOLS = {
    "claim_done": [tool_done],
    "python_execute": [tool_python_execute],
    "manage_context": context_management_tools,
    "history": history_tools,
    "handle_overlong_tool_outputs": overlong_tool_tools,
}


def limit_output(output: Any, context: Any) -> Any:
    text = output if isinstance(output, str) else str(output)
    limit = int(os.getenv("BENCH_MAX_SINGLE_TURN_RETURN_CHARS", "100000"))
    if len(text) <= limit:
        return output
    identifier = uuid.uuid4().hex
    path = Path(context.context["_agent_workspace"]) / ".overlong_tool_outputs" / f"{identifier}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    return (
        f"{text[:limit]} ...\n\n(The output of the tool call (shortuuid identifier: {identifier}) is too long! "
        f"The original output length is {len(text)} characters. The full output has been saved to {path}. "
        "Use the overlong-output tools to inspect it.)"
    )


def local_tools(names: list[str]) -> list[FunctionTool]:
    result = []
    for name in names:
        for original in LOCAL_TOOLS[name]:

            async def invoke(context, params, callback=original.on_invoke_tool):
                return limit_output(await callback(context, params), context)

            result.append(replace(original, name=to_model_tool_name(original.name), on_invoke_tool=invoke))
    return result


def coerce_arguments(arguments: Any, schema: dict[str, Any]) -> Any:
    """Preserve Toolathlon's acceptance of JSON-encoded structured MCP args."""
    if not isinstance(schema, dict):
        return arguments
    alternatives = [schema, *schema.get("anyOf", []), *schema.get("oneOf", [])]
    types = {
        kind
        for option in alternatives
        if isinstance(option, dict)
        for kind in (option.get("type", []) if isinstance(option.get("type"), list) else [option.get("type")])
    }
    if isinstance(arguments, str) and types & {"object", "array", "integer", "number", "boolean"}:
        try:
            decoded = json.loads(arguments)
        except ValueError:
            pass
        else:
            if (
                ("object" in types and isinstance(decoded, dict))
                or ("array" in types and isinstance(decoded, list))
                or ("boolean" in types and isinstance(decoded, bool))
                or ("number" in types and isinstance(decoded, (int, float)) and not isinstance(decoded, bool))
                or ("integer" in types and isinstance(decoded, int) and not isinstance(decoded, bool))
            ):
                arguments = decoded
    if isinstance(arguments, dict):
        properties = schema.get("properties", {})
        return {key: coerce_arguments(value, properties.get(key, {})) for key, value in arguments.items()}
    if isinstance(arguments, list):
        return [coerce_arguments(value, schema.get("items", {})) for value in arguments]
    return arguments


def mcp_tool(server: Any, tool: Any) -> FunctionTool:
    schema = dict(tool.inputSchema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})

    async def invoke(context, params):
        arguments = coerce_arguments(json.loads(params), schema)
        result = await server.call_tool(tool.name, arguments)
        span = get_current_span()
        if span is not None:
            span.span_data.mcp_data = {"server": server.name}
        content = [item.model_dump(mode="json") for item in result.content]
        output = json.dumps(content[0] if len(content) == 1 else content, ensure_ascii=False)
        return limit_output(output, context)

    return FunctionTool(
        name=to_model_mcp_tool_name(server.name, tool.name),
        description=tool.description or "",
        params_json_schema=schema,
        on_invoke_tool=invoke,
        strict_json_schema=False,
    )
