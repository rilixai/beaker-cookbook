"""One research request with OpenAI's built-in web search. Beaker may rewrite this file.

`model` is the pinned model with a per-question call cap and web-action budget. It is
fixed; everything here is the agent.
"""

from __future__ import annotations

import re

from livenewsbench.openai_search import ResponsesModel

from .prompts import FINAL_PROMPT, SYSTEM_PROMPT, USER_PROMPT


FINAL_ANSWER = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


async def answer_question(question: str, *, model: ResponsesModel) -> str:
    prompt = USER_PROMPT.format(question=question, max_web_actions=model.max_web_actions)
    reply = (await model.respond(prompt, instructions=SYSTEM_PROMPT)).output_text
    if FINAL_ANSWER.search(reply) or model.remaining_calls <= 0:
        return _final_answer(reply)
    # No answer line: ask once more for the answer, without search.
    follow_up = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": FINAL_PROMPT},
    ]
    return _final_answer((await model.respond(follow_up, instructions=SYSTEM_PROMPT, search=False)).output_text)


def _final_answer(text: str) -> str:
    match = FINAL_ANSWER.search(text)
    return (match.group(1) if match else text).strip()
