"""Application code: one user message, one structured chat call, one CLINC150 intent."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import ResponseFormatJSONSchema

from clinc150_router.contracts import LABELS, Prediction


class InvalidAnswer(ValueError):
    """The model answered, but not with exactly one known intent ID."""


class Span(Protocol):
    def output(self, value: Any, /) -> None: ...
    def usage(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None: ...


class Tracer(Protocol):
    """The part of ``beaker.tracing.Trace`` this app uses. Every model or decision call must be traced."""

    def model_call(
        self, *, operation: str, provider: str, model: str, input_messages: Any, server_address: str | None = None
    ) -> AbstractContextManager[Span]: ...


@dataclass(frozen=True)
class ChatModel:
    """``model`` is sent as-is; a gateway's ``<provider>:<model>`` name is traced without its prefix."""

    client: AsyncOpenAI
    provider: str
    model: str
    reasoning_effort: ReasoningEffort = None


def system_prompt(prompts_dir: Path) -> str:
    instructions = (prompts_dir / "instructions.md").read_text().strip()
    intents = json.loads((prompts_dir / "intents.json").read_text())
    if not instructions:
        raise ValueError("Routing instructions cannot be empty.")
    if not isinstance(intents, dict) or set(intents) != set(LABELS):
        raise ValueError("Keep all 151 intent IDs fixed; edit only their descriptions.")
    if not all(isinstance(value, str) and value.strip() for value in intents.values()):
        raise ValueError("Every intent needs a non-empty text description.")
    listing = "\n".join(f"- {intent}: {description.strip()}" for intent, description in intents.items())
    return f"{instructions}\n\nIntents:\n{listing}"


RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "route",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": list(LABELS)}},
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise InvalidAnswer("The model returned duplicate JSON keys.")
    return result


def parse_intent(content: str | None) -> str:
    try:
        answer = json.loads(content or "", object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise InvalidAnswer("The model did not return JSON.") from exc
    if not isinstance(answer, dict) or answer.keys() != {"intent"}:
        raise InvalidAnswer("The model must return an object with only an intent field.")
    intent = answer["intent"]
    if not isinstance(intent, str) or intent not in LABELS:
        raise InvalidAnswer("The model returned an unknown intent.")
    return intent


async def route(text: str, *, chat: ChatModel, prompt: str, trace: Tracer) -> Prediction:
    if not text.strip():
        raise ValueError("User message cannot be empty.")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    traced_model = chat.model.removeprefix(f"{chat.provider}:")
    with trace.model_call(
        operation="chat", provider=chat.provider, model=traced_model, input_messages=messages
    ) as call:
        response = await chat.client.chat.completions.create(
            model=chat.model,
            messages=messages,
            response_format=RESPONSE_FORMAT,
            reasoning_effort=chat.reasoning_effort or omit,
        )
        usage = response.usage
        choice = response.choices[0] if len(response.choices) == 1 else None
        content = choice.message.content if choice is not None else None
        call.output(content)
        call.usage(
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
    intent = None
    error = None
    try:
        if choice is None:
            raise InvalidAnswer("The model did not return exactly one choice.")
        if choice.message.refusal:
            raise InvalidAnswer("The model refused to answer.")
        if choice.finish_reason != "stop":
            raise InvalidAnswer("The model did not complete the answer.")
        intent = parse_intent(content)
    except InvalidAnswer as exc:
        error = str(exc)
    return Prediction(
        intent=intent,
        error=error,
        model=response.model,
        input_tokens=usage.prompt_tokens if usage else None,
        output_tokens=usage.completion_tokens if usage else None,
        cost_usd=getattr(usage, "cost", None),
    )
