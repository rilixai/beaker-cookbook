"""LiveNewsBench's grader: the SimpleQA grading prompt, answered by GPT-4.1 at temperature 0.

`grade_simple_qa_prompt.txt` is upstream `evals/grade_simple_qa_prompt.txt`, byte for
byte. Upstream counts a reply of exactly "A" as correct. This module keeps that rule
but retries a reply that is not A, B or C and then raises, so a judge failure never
turns into a silent wrong answer.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Literal

from openai import AsyncOpenAI


GRADER_MODEL = "gpt-4.1"
GRADER_MAX_TOKENS = 1024
GRADER_ATTEMPTS = 3
GRADE_TEMPLATE = files("livenewsbench").joinpath("grade_simple_qa_prompt.txt").read_text().strip()
Grade = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
LETTERS: dict[str, Grade] = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


class GraderError(RuntimeError):
    pass


def grader_prompt(question: str, gold: str, predicted: str) -> str:
    return GRADE_TEMPLATE.format(question=question, expected_answer=gold, answer=predicted)


async def grade(client: AsyncOpenAI, *, model: str, question: str, gold: str, predicted: str) -> Grade:
    if not predicted.strip():
        return "NOT_ATTEMPTED"
    replies = []
    for _ in range(GRADER_ATTEMPTS):
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": grader_prompt(question, gold, predicted)},
            ],
            max_completion_tokens=GRADER_MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
        )
        reply = (completion.choices[0].message.content or "").strip()
        if reply in LETTERS:
            return LETTERS[reply]
        replies.append(reply[:40])
    raise GraderError(f"Grader gave no A/B/C verdict in {GRADER_ATTEMPTS} attempts: {replies!r}")
