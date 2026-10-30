"""Application code: one customer message, one native TypeSafe Choice question."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator
from typesafe_sdk import AsyncTypeSafeClient, Choice


MODEL = "jev-1.14.0"
LABELS: tuple[str, ...] = tuple(json.loads(Path(__file__).with_name("categories.json").read_text()))


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    intent: str
    probabilities: dict[str, float]
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float

    @model_validator(mode="after")
    def valid_distribution(self) -> Prediction:
        if self.intent not in LABELS or set(self.probabilities) != set(LABELS):
            raise ValueError("The response must use exactly the 77 BANKING77 intent IDs.")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("Probabilities must be finite and in [0, 1].")
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=0.02):
            raise ValueError("Probabilities must sum to approximately one.")
        if self.probabilities[self.intent] < max(self.probabilities.values()):
            raise ValueError("Selected intent must have the highest probability.")
        return self


def question(prompts_dir: Path) -> Choice:
    instructions = (prompts_dir / "instructions.md").read_text().strip()
    criteria = json.loads((prompts_dir / "criteria.json").read_text())
    if not instructions:
        raise ValueError("Routing instructions cannot be empty.")
    if not isinstance(criteria, dict) or set(criteria) != set(LABELS):
        raise ValueError("Keep all 77 intent IDs fixed; edit only their descriptions.")
    if not all(isinstance(value, str) and value.strip() for value in criteria.values()):
        raise ValueError("Every intent needs a non-empty text description.")
    return Choice(instructions=instructions, criteria=criteria)


async def classify(text: str, *, client: AsyncTypeSafeClient, routing_question: Choice) -> Prediction:
    if not text.strip():
        raise ValueError("Customer message cannot be empty.")
    started = time.perf_counter()
    response = await client.system_one(state=text, questions={"intent": routing_question}, model=MODEL)
    answer = response.choices["intent"]
    return Prediction(
        intent=answer.choice,
        probabilities=answer.probabilities,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
