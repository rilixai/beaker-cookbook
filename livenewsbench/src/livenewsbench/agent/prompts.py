"""Prompts and tool descriptions for the web-search agent. Beaker may rewrite this file."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are a web search agent that performs web searches and reasoning to answer user questions. \
You are excellent at carefully following instructions and thinking step by step. \
You are also good at following output format instructions."""

USER_PROMPT = """Please perform online research to answer the following question:
{question}

# Tools
- `search`: search the web with a search engine. Keep queries concise and specific, as you would type them into \
Google. Results give each page's title, URL, date and a snippet.
- `open_page`: read the full content of a page. You can only open URLs returned by an earlier search.

# Budget
You can perform up to {max_searches} searches and {max_page_visits} page visits in total. Every search query and \
every page you open counts towards the budget, and calls over budget are refused. Try to minimize the number of \
searches and page visits needed before you answer.

# Answer
Think step by step before each tool call. When you have gathered enough information, reply without calling a tool, \
ending your reply with one line of the form:
Final answer: <your answer>
The final answer should be concise and complete, usually a phrase of a few words."""

FINAL_PROMPT = """You cannot call any more tools. Based on what you have found, reply with one line of the form:
Final answer: <your answer>"""

SEARCH_RESULT = """[{index}] {title}
URL: {url}
Published: {published}
{snippet}"""

BUDGET_NOTE = "You have used {searches} of {max_searches} searches and {page_visits} of {max_page_visits} page visits."

MAX_PAGE_CHARACTERS = 24_000

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web. Uses one search from the budget.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_page",
            "description": "Read the full content of a page returned by an earlier search. Uses one page visit.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "A URL from earlier search results."}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]
