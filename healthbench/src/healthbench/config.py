"""Runtime knobs shared by the CLI, the agent, and the grader."""

from __future__ import annotations

from dataclasses import dataclass


# The grader the HealthBench paper validated against physician agreement.
DEFAULT_GRADER_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_TASK_MODEL = "gpt-4.1-mini"


@dataclass(frozen=True)
class HealthBenchConfig:
    """Model + concurrency settings.

    ``task_model`` answers the conversation; ``grader_model`` judges each
    rubric criterion. Both go through the OpenAI chat-completions API; the SDK
    reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the environment, so
    pointing at OpenRouter or another compatible endpoint needs no code change.
    """

    task_model: str = DEFAULT_TASK_MODEL
    task_temperature: float = 0.0
    task_max_tokens: int = 2048
    grader_model: str = DEFAULT_GRADER_MODEL
    max_concurrency: int = 8
    # Retries for a judge reply that is not a ``{"criteria_met": bool}`` JSON.
    grader_max_retries: int = 3

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {self.max_concurrency}.")
        if self.grader_max_retries < 1:
            raise ValueError(f"grader_max_retries must be >= 1, got {self.grader_max_retries}.")
