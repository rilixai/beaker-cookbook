"""The customer-service agent: the code Beaker may edit.

A turn-based tau2 agent. Each turn it sends the instructions from
``prompts/system.md``, the domain policy and the conversation so far to the
chat model, and returns either a message to the user or tool calls.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import APICompatibleMessage, AssistantMessage, Message, MultiToolMessage, SystemMessage
from tau2.environment.tool import Tool

from tau2_bench.llm import ChatModel


SYSTEM_PROMPT = """<instructions>
{instructions}
</instructions>
<policy>
{policy}
</policy>"""


def load_instructions(prompts_dir: Path) -> str:
    instructions = (prompts_dir / "system.md").read_text().strip()
    if not instructions:
        raise ValueError("prompts/system.md cannot be empty.")
    return instructions


class AgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class Agent(HalfDuplexAgent[AgentState]):
    def __init__(self, tools: list[Tool], domain_policy: str, *, chat: ChatModel, instructions: str) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.chat = chat
        self.instructions = instructions

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(instructions=self.instructions, policy=self.domain_policy)

    def get_init_state(self, message_history: list[Message] | None = None) -> AgentState:
        return AgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        reply = self.chat.complete([*state.system_messages, *state.messages], self.tools)
        state.messages.append(reply)
        return reply, state
