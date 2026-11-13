"""Fixed intent labels, prediction schema, and seed model settings."""

import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator


LABELS: tuple[str, ...] = tuple(json.loads(Path(__file__).with_name("labels.json").read_text()))
OUT_OF_SCOPE = "oos"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT: Final = "low"
REQUEST_TIMEOUT_SECONDS = 60.0


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    intent: str | None
    error: str | None = None
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @field_validator("intent")
    @classmethod
    def known_intent(cls, value: str | None) -> str | None:
        if value is not None and value not in LABELS:
            raise ValueError("Unknown CLINC150 intent")
        return value
