"""GPT with OpenAI's built-in web search: a baseline against the You.com router and search.

OpenAI runs the search loop inside each request, so the benchmark's rules are applied
differently from the You.com tools in `tools.py`:

- Searches, page opens and in-page finds share one budget of MAX_WEB_ACTIONS built-in
  tool calls per question, the You.com side's 5 searches plus 5 page visits. Each request
  sets `max_tool_calls` to what is left.
- OpenAI's `filters` take only an allowlist, so the blocked hosts can't be excluded up
  front. Every URL the model searched, opened or cited is checked afterwards. When a
  question opened, searched in or cited a blocked host, its answer is withheld, so it
  scores as not attempted. Blocked hosts that only appeared among search results are
  counted.

In Beaker's sandbox, a client with no base URL or key reaches OpenAI through Beaker's
provider proxy, which meters tokens and searches. Beaker's gateway accepts built-in web
search only on that route.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from beaker.tracing import Trace
from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseFunctionWebSearch, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_output_text import AnnotationURLCitation

from livenewsbench.chat import CallLimitReached, ChatUsage, with_rate_limit_waits
from livenewsbench.tools import MAX_PAGE_VISITS, MAX_SEARCHES, blocked
from livenewsbench.youcom import TokenPrice


OPENAI_HOST = "api.openai.com"
DEFAULT_MODEL = "gpt-6-luna"
MAX_WEB_ACTIONS = MAX_SEARCHES + MAX_PAGE_VISITS
WEB_SEARCH_TOOL: dict[str, Any] = {"type": "web_search"}
# Return each search's result URLs, so they can be checked against the blocked hosts.
INCLUDE = ["web_search_call.action.sources"]
# Pinned for reproducible baselines; medium is the model's default.
REQUEST_OPTIONS: dict[str, Any] = {"reasoning": {"effort": "medium"}}
# OpenAI list prices as recorded in Beaker's rate table, checked 2026-11-12: USD per 1M
# tokens, and per search. Page opens and finds are not billed as searches.
OPENAI_PRICES = {"gpt-6-luna": TokenPrice(0.10, 0.01, 0.50)}
OPENAI_SEARCH_PRICE = 10.00 / 1000


def openai_client() -> AsyncOpenAI:
    """OpenAI's own endpoint; Beaker's sandbox routes it through Beaker's provider proxy."""
    return AsyncOpenAI(max_retries=5, timeout=900)


@dataclass
class OpenAIWebUsage:
    searches: int = 0
    page_visits: int = 0
    finds: int = 0
    blocked_in_results: int = 0
    blocked_used: list[str] = field(default_factory=list)

    @property
    def actions(self) -> int:
        return self.searches + self.page_visits + self.finds


@dataclass
class ResponsesModel:
    """One question's model access: Responses with built-in web search, capped and audited."""

    client: AsyncOpenAI
    model: str
    max_calls: int
    trace: Trace
    max_web_actions: int = MAX_WEB_ACTIONS
    request_options: Mapping[str, Any] = field(default_factory=lambda: dict(REQUEST_OPTIONS))
    usage: ChatUsage = field(default_factory=ChatUsage)
    web: OpenAIWebUsage = field(default_factory=OpenAIWebUsage)

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.usage.calls

    @property
    def remaining_web_actions(self) -> int:
        return max(0, self.max_web_actions - self.web.actions)

    async def respond(
        self, input: str | list[dict[str, Any]], *, instructions: str | None = None, search: bool = True
    ) -> Response:
        """One request; `search` offers the web search tool while the question's budget lasts."""
        if self.remaining_calls <= 0:
            raise CallLimitReached(f"The agent may make at most {self.max_calls} model calls per question")
        request: dict[str, Any] = {"model": self.model, "input": input, **self.request_options}
        if instructions is not None:
            request["instructions"] = instructions
        if search and self.remaining_web_actions > 0:
            request |= {"tools": [WEB_SEARCH_TOOL], "max_tool_calls": self.remaining_web_actions, "include": INCLUDE}
        self.usage.calls += 1
        with self.trace.model_call(
            operation="responses",
            provider="openai",
            model=self.model,
            server_address=OPENAI_HOST,
            input_messages=input,
        ) as call:
            started = time.perf_counter()
            try:
                response: Response = await with_rate_limit_waits(
                    lambda: self.client.responses.create(**request), self.usage, "openai response"
                )
            finally:
                self.usage.seconds += time.perf_counter() - started
            call.output(response)
        self._meter(response)
        return response

    def _meter(self, response: Response) -> None:
        if response.usage is not None:
            self.usage.input_tokens += response.usage.input_tokens
            self.usage.output_tokens += response.usage.output_tokens
            self.usage.cached_input_tokens += response.usage.input_tokens_details.cached_tokens or 0
            self.usage.reasoning_tokens += response.usage.output_tokens_details.reasoning_tokens or 0
        for item in response.output:
            if isinstance(item, ResponseFunctionWebSearch):
                self._meter_web_action(item)
            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if isinstance(part, ResponseOutputText):
                        for annotation in part.annotations:
                            if isinstance(annotation, AnnotationURLCitation) and blocked(annotation.url):
                                self.web.blocked_used.append(annotation.url)

    def _meter_web_action(self, item: ResponseFunctionWebSearch) -> None:
        action = item.action
        # Replayed as tool spans after the request, so their timings are not real.
        with self.trace.tool_call("openai_web_search", arguments=action.model_dump(exclude_none=True)) as span:
            span.output({"status": item.status})
        if isinstance(action, ActionSearch):
            self.web.searches += 1
            self.web.blocked_in_results += sum(blocked(source.url) for source in action.sources or [])
        elif isinstance(action, ActionOpenPage):
            self.web.page_visits += 1
            if action.url and blocked(action.url):
                self.web.blocked_used.append(action.url)
        elif isinstance(action, ActionFind):
            self.web.finds += 1
            if blocked(action.url):
                self.web.blocked_used.append(action.url)
