"""Route one user message with one native TypeSafe Choice question."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol

from typesafe_sdk import AsyncTypeSafeClient, Choice

from clinc150_router.contracts import DEFAULT_MODEL, LABELS, Prediction


class Span(Protocol):
    def output(self, value: Any, /) -> None: ...
    def usage(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None: ...


class Tracer(Protocol):
    def model_call(
        self, *, operation: str, provider: str, model: str, input_messages: Any, server_address: str | None = None
    ) -> AbstractContextManager[Span]: ...


def question(prompts_dir: Path) -> Choice:
    instructions = (prompts_dir / "instructions.md").read_text().strip()
    intents = json.loads((prompts_dir / "intents.json").read_text())
    if not instructions:
        raise ValueError("Routing instructions cannot be empty.")
    if not isinstance(intents, dict) or set(intents) != set(LABELS):
        raise ValueError("Keep all 151 intent IDs fixed; edit only their descriptions.")
    if not all(isinstance(value, str) and value.strip() for value in intents.values()):
        raise ValueError("Every intent needs a non-empty text description.")
    return Choice(
        instructions=instructions, criteria={intent: description.strip() for intent, description in intents.items()}
    )


async def route(text: str, *, client: AsyncTypeSafeClient, routing_question: Choice, trace: Tracer) -> Prediction:
    if not text.strip():
        raise ValueError("User message cannot be empty.")
    with trace.model_call(
        operation="systemone",
        provider="typesafe",
        model=DEFAULT_MODEL,
        server_address="api.typesafe.ai",
        input_messages={"state": text, "questions": {"intent": routing_question.model_dump(mode="json")}},
    ) as call:
        response = await client.system_one(state=text, questions={"intent": routing_question}, model=DEFAULT_MODEL)
        call.output(response.model_dump(mode="json"))
        call.usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

    answer = response.choices.get("intent")
    intent = answer.choice if answer is not None else None
    return Prediction(
        intent=intent if intent in LABELS else None,
        error=None if intent in LABELS else "The decision model did not return a known intent.",
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
