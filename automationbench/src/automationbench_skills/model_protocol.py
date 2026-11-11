"""Recipe-owned model messages and provider adapters.

PI no longer exports the legacy Verifiers client API. Keep the recipe's
provider routing independent of the benchmark's choice of rollout framework.
"""

from __future__ import annotations

import json
import os
from typing import Any, Generic, Literal, TypeVar

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ConfigDict, Field


class Error(Exception):
    """A rollout failure."""


class ModelError(Error):
    """A provider request failed after retries."""


class InfraError(Error):
    """Infrastructure failed before the agent could finish."""


class OverlongPromptError(Error):
    """The candidate exceeded the model context limit."""


class ClientConfig(BaseModel):
    api_key_var: str = "OPENAI_API_KEY"
    api_base_url: str = "https://api.openai.com/v1"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    max_retries: int = 0
    timeout: float = 300.0
    connect_timeout: float = 30.0
    max_connections: int = 1000
    max_keepalive_connections: int = 100


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = None


class SystemMessage(Message):
    role: Literal["system"] = "system"


class UserMessage(Message):
    role: Literal["user"] = "user"


TextMessage = UserMessage


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    anthropic_content: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    is_truncated: bool = False


ResponseMessage = AssistantMessage


class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    tool_call_id: str


class Tool(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class Response(BaseModel):
    id: str
    created: int
    model: str
    usage: Usage | None = None
    message: ResponseMessage


Messages = list[Message]
SamplingArgs = dict[str, Any]
FinishReason = str


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")


class Client(Generic[A, B, C, D]):
    """Small protocol shared by scripted clients and provider adapters."""

    def __init__(self, client_or_config: Any) -> None:
        self._client = (
            self.setup_client(client_or_config) if isinstance(client_or_config, ClientConfig) else client_or_config
        )

    @property
    def client(self) -> Any:
        return self._client

    def setup_client(self, config: ClientConfig) -> Any:
        raise NotImplementedError

    async def to_native_prompt(self, messages: Messages) -> Any:
        raise NotImplementedError

    async def to_native_tool(self, tool: Tool) -> Any:
        raise NotImplementedError

    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def from_native_response(self, response: Any) -> Response:
        raise NotImplementedError

    async def raise_from_native_response(self, response: Any) -> None:
        return None

    async def close(self) -> None:
        await self.client.close()


def setup_openai_client(config: ClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ[config.api_key_var],
        base_url=config.api_base_url,
        default_headers=config.extra_headers,
        max_retries=config.max_retries,
        timeout=config.timeout,
    )


def parse_reasoning_content(message: Any) -> str | None:
    return getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)


class OpenAIChatCompletionsClient(Client):
    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def to_native_tool(self, tool: Tool) -> dict[str, Any]:
        return {"type": "function", "function": tool.model_dump()}

    async def to_native_prompt(self, messages: Messages) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out = []
        for message in messages:
            row = {"role": message.role, "content": message.content}
            if isinstance(message, AssistantMessage):
                if message.tool_calls:
                    row["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                        for c in message.tool_calls
                    ]
                if message.reasoning_content is not None:
                    row["reasoning_content"] = message.reasoning_content
            if isinstance(message, ToolMessage):
                row["tool_call_id"] = message.tool_call_id
            out.append(row)
        return out, {}

    async def get_native_response(
        self, prompt: Any, model: str, sampling_args: Any, tools: Any = None, **kwargs: Any
    ) -> Any:
        kwargs.pop("state", None)
        try:
            return await self.client.chat.completions.create(
                model=model,
                messages=prompt,
                **({"tools": tools} if tools else {}),
                **sampling_args,
                **kwargs,
            )
        except BadRequestError as exc:
            # Preserve the legacy client's distinction between candidate context
            # overflow and provider failures, including compatible gateway responses.
            error_text = exc.response.text.lower()
            if any(
                marker in error_text
                for marker in (
                    "context_length_exceeded",
                    "prompt_too_long",
                    "context length",
                    "maximum model length",
                    "prompt is too long",
                )
            ):
                raise OverlongPromptError(str(exc)) from exc
            raise

    async def raise_from_native_response(self, response: Any) -> None:
        choices = getattr(response, "choices", None)
        if not choices or len(choices) != 1:
            raise ModelError("Model must return exactly one response choice")
        message = choices[0].message
        details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
        if not (
            message.content
            or message.tool_calls
            or getattr(message, "refusal", None)
            or parse_reasoning_content(message)
            or getattr(details, "reasoning_tokens", 0)
        ):
            raise ModelError("Model returned no content and did not call any tools")

    async def from_native_response(self, response: Any) -> Response:
        choice = response.choices[0]
        msg = choice.message
        usage = response.usage
        return Response(
            id=response.id,
            created=response.created,
            model=response.model,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None,
            message=ResponseMessage(
                content=msg.content or getattr(msg, "refusal", None),
                reasoning_content=parse_reasoning_content(msg),
                tool_calls=[
                    ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
                    for c in msg.tool_calls or []
                ]
                or None,
                finish_reason=choice.finish_reason,
                is_truncated=choice.finish_reason == "length",
            ),
        )


class AnthropicMessagesClient(Client):
    def setup_client(self, config: ClientConfig) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=os.environ[config.api_key_var], max_retries=config.max_retries, timeout=config.timeout
        )

    async def to_native_tool(self, tool: Tool) -> dict[str, Any]:
        return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}

    async def to_native_prompt(self, messages: Messages) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system = []
        content: Any
        rows: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                system.append(str(message.content))
                continue
            if isinstance(message, ToolMessage):
                role = "user"
                content = [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}]
            elif isinstance(message, AssistantMessage):
                role = "assistant"
                content = getattr(message, "anthropic_content", None)
                if content is None:
                    content = [{"type": "text", "text": message.content}] if message.content else []
                    content += [
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": json.loads(c.arguments)}
                        for c in message.tool_calls or []
                    ]
            else:
                role = "user"
                content = (
                    [{"type": "text", "text": message.content}]
                    if isinstance(message.content, str)
                    else message.content
                )
            if rows and rows[-1]["role"] == role:
                rows[-1]["content"].extend(content)
            else:
                rows.append({"role": role, "content": content})
        return rows, {"system": "\n\n".join(system)} if system else {}

    async def from_native_response(self, response: Any) -> Response:
        blocks = [b.model_dump(mode="json") for b in response.content]
        usage = response.usage
        input_tokens = (
            usage.input_tokens
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        )
        return Response(
            id=response.id,
            created=0,
            model=response.model,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=input_tokens + usage.output_tokens,
            ),
            message=ResponseMessage(
                content="".join(b["text"] for b in blocks if b["type"] == "text"),
                tool_calls=[
                    ToolCall(id=b["id"], name=b["name"], arguments=json.dumps(b["input"]))
                    for b in blocks
                    if b["type"] == "tool_use"
                ]
                or None,
                anthropic_content=blocks,
                finish_reason=response.stop_reason,
                is_truncated=response.stop_reason == "max_tokens",
            ),
        )
