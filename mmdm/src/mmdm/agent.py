"""The agent: answers one MMDM question with a decision model. Editable.

``answer`` receives the question (state, question, options, task and score
values; never labels or category) and a ``Session`` whose ``decide`` sends a
request to the decision model and whose ``chat`` calls a chat model. It
returns a probability for every original option.

Seed behavior: one native decision call per question. ``prompts/instructions.md``
goes in front of the question's own instructions, and a state too long for the
request budget is cut with upstream head-tail truncation.
"""

from __future__ import annotations

from pathlib import Path

from mmdm.data import Question
from mmdm.decide import Session, head_tail


def load_instructions(prompts_dir: Path) -> str:
    return (prompts_dir / "instructions.md").read_text().strip()


async def answer(question: Question, session: Session, instructions: str) -> dict[str, float]:
    text = f"{instructions}\n\n{question.question}" if instructions else question.question
    request, _ = head_tail(
        question.model_copy(update={"question": text}), model=session.target.model, counter=session.counter
    )
    decision = await session.decide(request)
    return decision.probabilities
