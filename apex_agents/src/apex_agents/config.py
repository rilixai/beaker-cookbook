"""Runtime configuration for the APEX-Agents benchmark.

``ApexAgentsConfig`` is the single knob bag shared by the CLI and the
evaluation package. Lives at the top level (peer of ``cli.py``) so
everything can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


# Kept inline (not imported from ``evaluation.scoring``) so this module
# sits at the very bottom of the package import graph. The canonical
# definition lives in ``evaluation.scoring``; this mirror is kept in
# lockstep.
DEFAULT_JUDGE_MODEL = "gemini/gemini-3.5-flash"


@dataclass(frozen=True)
class ApexAgentsConfig:
    """Knobs for an APEX-Agents run.

    ``task_model`` is the LiteLLM model spec for the ReAct agent;
    ``judge_model`` is the LLM-judge model (Mercor default
    ``gemini/gemini-3.5-flash``). ``max_steps`` / ``cost_limit`` cap
    the inner ReAct loop (``max_steps`` defaults to 60 — smaller than
    Archipelago's 250 to bound demo cost).
    """

    task_model: str = "openai/gpt-4.1-mini-2025-04-14"
    task_temperature: float = 0.0
    judge_model: str = DEFAULT_JUDGE_MODEL
    max_steps: int = 60
    cost_limit: float = 3.0
    max_toolbelt_size: int = 80
    max_context_tokens: int = 120_000
    # Per-LLM-call timeout (seconds) for the agent model AND the rubric
    # judge. Bounds a hung provider request so the case fails fast
    # instead of wedging the whole run.
    llm_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"ApexAgentsConfig.max_steps must be >= 1, got {self.max_steps}.")
        if self.cost_limit <= 0:
            raise ValueError(f"ApexAgentsConfig.cost_limit must be > 0, got {self.cost_limit}.")
        if self.max_toolbelt_size < 1:
            raise ValueError(f"ApexAgentsConfig.max_toolbelt_size must be >= 1, got {self.max_toolbelt_size}.")
        if self.llm_timeout <= 0:
            raise ValueError(f"ApexAgentsConfig.llm_timeout must be > 0, got {self.llm_timeout}.")
