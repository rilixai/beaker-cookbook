"""The HealthBench agent: one chat-completions call per conversation.

HealthBench is single-response: the conversation so far is the input and the
model's next assistant turn is the output. The agent prepends its system
prompt and returns that turn. It is deliberately minimal so prompt and model
changes are the obvious levers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ..data.dataset import HealthBenchRecord
from .prompts import SYSTEM_PROMPT


@dataclass
class AgentOutput:
    response: str
    usage: dict[str, int] = field(default_factory=dict)


class HealthBenchAgent:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    async def respond(self, record: HealthBenchRecord) -> AgentOutput:
        messages = [{"role": "system", "content": self.system_prompt}, *record.prompt]
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = completion.usage.model_dump(include={"prompt_tokens", "completion_tokens"}) if completion.usage else {}
        return AgentOutput(response=completion.choices[0].message.content or "", usage=usage)
