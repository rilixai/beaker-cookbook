"""The agent: answers one DecisionPivot item with a decision model. Editable.

``answer`` receives one item (its id, state and question; never the answer,
level or the other members of its contrast group) and a ``Session`` whose
``decide`` sends a request to the decision model and whose ``chat`` calls a
chat model. It returns a probability for every option of the item.

Seed behavior: one native decision call per item with the item's state and
question. A non-empty ``prompts/instructions.md`` goes in front of the
question's own instructions.
"""

from __future__ import annotations

from pathlib import Path

from decisionpivot.data import Item
from decisionpivot.decide import Request, Session


def load_instructions(prompts_dir: Path) -> str:
    return (prompts_dir / "instructions.md").read_text().strip()


async def answer(item: Item, session: Session, instructions: str) -> dict[str, float]:
    request = Request.of(item)
    if instructions:
        question = item.question
        text = f"{instructions}\n\n{question.instructions}"
        request = Request(state=item.state, question=question.model_copy(update={"instructions": text}))
    decision = await session.decide(request)
    return decision.probabilities
