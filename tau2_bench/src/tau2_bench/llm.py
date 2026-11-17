"""Use native Jev decisions to route turns and Chat Completions to generate them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from beaker.sdk.utils import to_json_safe
from beaker.tracing import current_trace
from openai import OpenAI
from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import to_litellm_messages
from typesafe_sdk import Choice, TypeSafeClient


DECISION_MODEL = "jev-1.14.0"
REPLY = "reply_to_customer"
DECISION_INSTRUCTIONS = (
    "Choose the next action for the customer-service agent. Follow the policy in the system message. "
    "Choose a tool only when its prerequisites are met and the conversation contains enough information "
    "to call it. Choose a reply when you need information or confirmation from the customer, or when "
    "you can answer them from the information already available."
)


@dataclass(frozen=True)
class ChatModel:
    client: OpenAI
    model: str
    reasoning_effort: str | None = None
    decisions: TypeSafeClient = field(default_factory=lambda: TypeSafeClient(timeout=60))

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            self.decisions.close()

    def choose_action(self, messages: list[dict[str, Any]], tools: dict[str, dict[str, Any]]) -> str:
        criteria = {name: schema["function"]["description"] for name, schema in tools.items()}
        criteria[REPLY] = "Send a message, ask a question, request confirmation, or explain the outcome."
        question = Choice(instructions=DECISION_INSTRUCTIONS, criteria=criteria)
        with current_trace().model_call(
            operation="systemone",
            provider="typesafe",
            model=DECISION_MODEL,
            server_address="api.typesafe.ai",
            input_messages={"state": messages, "questions": {"next_action": question.model_dump(mode="json")}},
        ) as call:
            decision = self.decisions.system_one(
                state=messages, questions={"next_action": question}, model=DECISION_MODEL
            )
            call.output(decision.model_dump(mode="json"))
            call.usage(input_tokens=decision.usage.input_tokens, output_tokens=decision.usage.output_tokens)
        answer = decision.choices.get("next_action")
        if answer is None or answer.choice not in criteria:
            raise ValueError("Jev returned an invalid next action.")
        return answer.choice

    def complete(self, messages: list[Message], tools: list[Tool]) -> AssistantMessage:
        history = to_litellm_messages(messages)
        schemas = {tool.name: tool.openai_schema for tool in tools}
        if REPLY in schemas:
            raise ValueError(f"Tool name is reserved for the conversation router: {REPLY}")
        action = self.choose_action(history, schemas) if schemas else REPLY
        request: dict[str, Any] = {"model": self.model, "messages": history}
        if action != REPLY:
            request["tools"] = [schemas[action]]
            request["tool_choice"] = "required"
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
