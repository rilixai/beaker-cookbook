"""Hermetic test helpers: a scripted verifiers Client that never touches the
network. Each ScriptedClient instance replays a fixed sequence of assistant
turns (optionally with tool calls); the rollout, tool execution, state reset,
and rubric scoring all run for real in-process."""

from __future__ import annotations

import json
import time
from typing import Any

from verifiers.clients import Client
from verifiers.legacy.types import Response, ResponseMessage, ToolCall, Usage


class ScriptedClient(Client):
    """Replays scripted assistant turns; appends a final plain 'done' turn if
    the script runs out."""

    def __init__(self, turns: list[dict[str, Any]] | None = None) -> None:
        super().__init__(client_or_config=object())
        self.turns = list(turns or [])
        self.calls: list[Any] = []

    def setup_client(self, config: Any) -> Any:
        return object()

    async def to_native_tool(self, tool: Any) -> Any:
        return tool

    async def to_native_prompt(self, messages: Any) -> tuple[Any, dict]:
        return messages, {}

    async def raise_from_native_response(self, response: Any) -> None:
        return None

    async def from_native_response(self, response: Any) -> Response:
        return response

    async def close(self) -> None:
        return None

    async def get_native_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> Response:
        self.calls.append({"prompt": prompt, "tools": tools, "sampling_args": sampling_args})
        turn = self.turns.pop(0) if self.turns else {"content": "done"}
        tool_calls = [
            ToolCall(id=f"call_{i}", name=tc["name"], arguments=json.dumps(tc.get("arguments", {})))
            for i, tc in enumerate(turn.get("tool_calls", []))
        ]
        message = ResponseMessage(
            content=turn.get("content"),
            tool_calls=tool_calls or None,
            finish_reason="tool_calls" if tool_calls else "stop",
            is_truncated=False,
        )
        return Response(
            id=f"scripted-{len(self.calls)}",
            created=int(time.time()),
            model=model,
            usage=Usage(prompt_tokens=1, reasoning_tokens=0, completion_tokens=1, total_tokens=2),
            message=message,
        )
