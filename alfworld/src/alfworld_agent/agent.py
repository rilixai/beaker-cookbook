"""The agent: picks each ALFWorld action with a decision model. Editable.

``Agent.act`` receives a ``View`` (the task goal, the current observation, the
history so far and the admissible commands; never ``won``, the game file,
PDDL facts, the expert plan or the task type) and a ``Session`` whose
``decide`` asks the decision model a Choice question and whose ``chat`` calls
a chat model. It returns the next command, which must be admissible.

A new ``Agent`` is made for each episode, so it may keep its own memory across
steps. Decision requests may offer only admissible commands, but may leave
some out, and the agent may ask more than one question per step.

Seed behavior: one decision call per step. The state is the goal, the last
``HISTORY_STEPS`` steps and the current observation; the instructions are
``prompts/instructions.md``; every admissible command is an option, its text
both the key and the description.
"""

from __future__ import annotations

from pathlib import Path

from alfworld_agent.decide import Session
from alfworld_agent.env import View


HISTORY_STEPS = 10


def load_instructions(prompts_dir: Path) -> str:
    return (prompts_dir / "instructions.md").read_text().strip()


def render_state(view: View) -> str:
    lines = [f"Task: {view.goal}", ""]
    recent = view.history[-HISTORY_STEPS:]
    if recent:
        lines.append("Recent steps (oldest first):")
        first = len(view.history) - len(recent) + 1
        for number, step in enumerate(recent, start=first):
            lines.append(f"{number}. > {step.action}\n{step.observation}")
        lines.append("")
    lines.append(f"Current observation:\n{view.observation}")
    return "\n".join(lines)


class Agent:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions

    async def act(self, view: View, session: Session) -> str:
        decision = await session.decide(
            state=render_state(view),
            instructions=self.instructions,
            criteria={command: command for command in view.admissible_commands},
        )
        return decision.choice
