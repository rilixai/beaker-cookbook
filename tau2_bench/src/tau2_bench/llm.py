"""The one place the agent makes chat calls.

Every agent turn goes through ``ChatModel.complete``: an OpenAI-compatible Chat
Completions request, traced as one Beaker model call. Model-comparison runs
swap the model here (the integration builds the ``ChatModel``). A decision call
added beside it must be traced too, with provider ``typesafe`` or ``openai`` and
its operation (see the README), so baseline traces show where decisions happen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from beaker.sdk.utils import to_json_safe
from beaker.tracing import current_trace
from openai import OpenAI
from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import to_litellm_messages


@dataclass(frozen=True)
class ChatModel:
    client: OpenAI
    model: str
    reasoning_effort: str | None = None

    def complete(self, messages: list[Message], tools: list[Tool]) -> AssistantMessage:
        request: dict[str, Any] = {
            "model": self.model,
            # tau2's converter emits standard OpenAI chat messages.
            "messages": to_litellm_messages(messages),
            "tools": [tool.openai_schema for tool in tools] or None,
        }
        if request["tools"]:
            request["tool_choice"] = "auto"
        else:
            del request["tools"]
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        with current_trace().model_call(
            operation="chat.completions",
            provider="openai",
            model=self.model,
            input_messages=request["messages"],
            server_address=self.client.base_url.host,
        ) as call:
            response = self.client.chat.completions.create(**request)
            raw = response.model_dump(mode="json")
            call.output(to_json_safe(raw))
        choice = response.choices[0].message
        tool_calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=json.loads(call.function.arguments or "{}"))
            for call in choice.tool_calls or []
            if call.type == "function"
        ]
        usage = response.usage
        return AssistantMessage(
            role="assistant",
            content=choice.content,
            tool_calls=tool_calls or None,
            usage=None
            if usage is None
            else {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
            raw_data=raw,
        )
