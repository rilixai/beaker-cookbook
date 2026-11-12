"""A tool-calling web-search loop. Beaker may rewrite this file.

`chat` is the pinned model with a per-question call cap, and `web` enforces the
benchmark's search budget and exclusions. Both are fixed; everything here is the agent.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCallUnion

from livenewsbench.chat import ChatModel
from livenewsbench.tools import ToolRefused, WebTools

from .prompts import (
    BUDGET_NOTE,
    FINAL_PROMPT,
    MAX_PAGE_CHARACTERS,
    SEARCH_RESULT,
    SYSTEM_PROMPT,
    TOOLS,
    USER_PROMPT,
)


FINAL_ANSWER = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


async def answer_question(question: str, *, chat: ChatModel, web: WebTools) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                question=question, max_searches=web.max_searches, max_page_visits=web.max_page_visits
            ),
        },
    ]
    # Keep the last allowed model call for the answer.
    while chat.remaining_calls > 1:
        completion = await chat.complete(messages, tools=TOOLS)
        message = completion.choices[0].message
        messages.append(_assistant_message(message))
        if not message.tool_calls:
            return _final_answer(message.content)
        for tool_call in message.tool_calls:
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": await _run_tool(tool_call, web)})
    messages.append({"role": "user", "content": FINAL_PROMPT})
    completion = await chat.complete(messages)
    return _final_answer(completion.choices[0].message.content)


def _assistant_message(message: ChatCompletionMessage) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        turn["tool_calls"] = [tool_call.model_dump(exclude_none=True) for tool_call in message.tool_calls]
        # Thinking models such as DeepSeek's expect their reasoning back on tool-call turns.
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            turn["reasoning_content"] = reasoning
    else:
        turn["content"] = message.content or ""
    return turn


def _final_answer(content: str | None) -> str:
    text = (content or "").strip()
    match = FINAL_ANSWER.search(text)
    return (match.group(1) if match else text).strip()


async def _run_tool(tool_call: ChatCompletionMessageToolCallUnion, web: WebTools) -> str:
    if tool_call.type != "function":
        return f"Error: unsupported tool call type {tool_call.type!r}."
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
        if tool_call.function.name == "search":
            hits = await web.search(str(arguments["query"]))
            body = "\n\n".join(
                SEARCH_RESULT.format(
                    index=index,
                    title=hit.title,
                    url=hit.url,
                    published=hit.published or "unknown",
                    snippet=hit.snippet,
                )
                for index, hit in enumerate(hits, start=1)
            )
            result = body or "No results."
        elif tool_call.function.name == "open_page":
            page = await web.open_page(str(arguments["url"]))
            result = f"Title: {page.title}\nURL: {page.url}\n\n{page.content[:MAX_PAGE_CHARACTERS]}"
        else:
            result = f"Error: unknown tool {tool_call.function.name!r}."
    except ToolRefused as refusal:
        result = f"Refused: {refusal}"
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        result = f"Error: invalid arguments ({type(exc).__name__})."
    return result + "\n\n" + _budget(web)


def _budget(web: WebTools) -> str:
    return BUDGET_NOTE.format(
        searches=web.usage.searches,
        max_searches=web.max_searches,
        page_visits=web.usage.page_visits,
        max_page_visits=web.max_page_visits,
    )
