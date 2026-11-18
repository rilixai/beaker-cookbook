"""Terminus 2 seed with editable prompts and agent behavior.

The upstream agent supplies the terminal loop, JSON parser, tmux interaction,
and context summarization. Override its methods here to optimize that behavior.
The benchmark runner, task files, resource limits, and verifier are fixed.
"""

from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class CookbookTerminus(Terminus2):
    @staticmethod
    def name() -> str:
        return "cookbook-terminus-2"

    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("max_turns", 32)
        kwargs.setdefault("reasoning_effort", "none")
        kwargs.setdefault("record_terminal_session", False)
        kwargs.setdefault("llm_call_kwargs", {"max_tokens": 8192})
        super().__init__(*args, **kwargs)

    def _get_prompt_template_path(self) -> Path:
        return Path(__file__).parent / "prompts" / "system.txt"

    def _get_timeout_template_path(self) -> Path:
        return Path(__file__).parent / "prompts" / "timeout.txt"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        """Entry point for editable planning and terminal behavior."""
        await super().run(instruction, environment, context)
