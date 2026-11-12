"""The benchmark's web rules, enforced outside the editable agent.

LiveNewsBench allows 5 searches and 5 page visits per question by default, and
excludes Internet archive sites. A page may only be opened if a search in the same
question returned it. A call over budget, on a blocked domain, or on an unseen URL is
refused before any request is made, so a candidate agent cannot break the rules by
asking. When the web service itself rejects a request, the agent is told so instead
of the question failing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, NoReturn, Protocol
from urllib.parse import urlsplit

from beaker.tracing import Trace


MAX_SEARCHES = 5
MAX_PAGE_VISITS = 5
# The archive sites upstream's reference agent excludes, plus hosts that publish this
# benchmark's questions with their answers (dataset, code, paper, eval reports).
BLOCKED_DOMAINS = (
    "archive.is",
    "archive.ph",
    "archive.today",
    "archive.org",
    "waybackmachine.org",
    "huggingface.co",
    "hf.co",
    "github.com",
    "githubusercontent.com",
    "github.io",
    "arxiv.org",
    "braintrust.dev",
)
# The search service's query limit (You.com Search answers HTTP 422 above it). Checked
# here so an over-long query is refused before it is sent and uses no search.
MAX_QUERY_WORDS = 50
MAX_QUERY_CHARACTERS = 400


class ToolRefused(Exception):
    """The agent's request was not carried out, and the message says why.

    Either it breaks a rule (nothing was sent and no budget was used) or the web service
    rejected it (the search or page visit counts against the budget).
    """


class RequestRejected(Exception):
    """The web service refused this request itself, for example a malformed query or URL."""


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str
    published: str | None = None


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    content: str


SearchExtraction = Literal["snippets", "highlights"]


class SearchBackend(Protocol):
    async def search(
        self, query: str, *, count: int, extraction: SearchExtraction, exclude_domains: tuple[str, ...]
    ) -> list[SearchHit]: ...

    async def fetch(self, url: str) -> Page: ...


def blocked(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in BLOCKED_DOMAINS)


@dataclass
class WebUsage:
    searches: int = 0
    page_visits: int = 0
    search_seconds: float = 0.0
    page_seconds: float = 0.0
    refused: int = 0
    rejected: int = 0


@dataclass
class WebTools:
    """One question's web access."""

    backend: SearchBackend
    trace: Trace
    max_searches: int = MAX_SEARCHES
    max_page_visits: int = MAX_PAGE_VISITS
    usage: WebUsage = field(default_factory=WebUsage)
    _seen_urls: set[str] = field(default_factory=set)

    @property
    def remaining_searches(self) -> int:
        return self.max_searches - self.usage.searches

    @property
    def remaining_page_visits(self) -> int:
        return self.max_page_visits - self.usage.page_visits

    async def search(
        self, query: str, *, count: int = 10, extraction: SearchExtraction = "snippets"
    ) -> list[SearchHit]:
        """Search the web; each call uses one search from the budget.

        `count` is 1 to 20 results. `extraction` picks the text shown per result: the
        search engine's snippets, or query-relevant highlights from the page (same price).
        """
        query = query.strip()
        arguments = {"query": query, "count": count, "extraction": extraction}
        with self.trace.tool_call("web_search", arguments=arguments) as call:
            if not query:
                self._refuse("The search query is empty.")
            words = len(query.split())
            if words > MAX_QUERY_WORDS or len(query) > MAX_QUERY_CHARACTERS:
                self._refuse(
                    f"The search query is too long ({words} words, {len(query)} characters). Use at most "
                    f"{MAX_QUERY_WORDS} words and {MAX_QUERY_CHARACTERS} characters; no search was used."
                )
            if self.remaining_searches <= 0:
                self._refuse(f"The search budget of {self.max_searches} is used up.")
            if not 1 <= count <= 20:
                self._refuse("count must be between 1 and 20.")
            if extraction not in ("snippets", "highlights"):
                self._refuse("extraction must be 'snippets' or 'highlights'.")
            self.usage.searches += 1
            started = time.perf_counter()
            try:
                hits = await self.backend.search(
                    query, count=count, extraction=extraction, exclude_domains=BLOCKED_DOMAINS
                )
            except RequestRejected as exc:
                self._reject(f"The search service rejected this query ({exc}); the search was used.")
            finally:
                self.usage.search_seconds += time.perf_counter() - started
            hits = [hit for hit in hits if not blocked(hit.url)]
            self._seen_urls.update(hit.url for hit in hits)
            call.output([{"url": hit.url, "title": hit.title} for hit in hits])
        return hits

    async def open_page(self, url: str) -> Page:
        """Read a page a search returned; each call uses one page visit from the budget."""
        url = url.strip()
        with self.trace.tool_call("open_page", arguments={"url": url}) as call:
            if blocked(url):
                self._refuse("This domain is excluded by the benchmark rules.")
            if url not in self._seen_urls:
                self._refuse("Only URLs returned by an earlier search in this question can be opened.")
            if self.remaining_page_visits <= 0:
                self._refuse(f"The page-visit budget of {self.max_page_visits} is used up.")
            self.usage.page_visits += 1
            started = time.perf_counter()
            try:
                page = await self.backend.fetch(url)
            except RequestRejected as exc:
                self._reject(f"The page could not be fetched ({exc}); the page visit was used.")
            finally:
                self.usage.page_seconds += time.perf_counter() - started
            call.output({"url": page.url, "title": page.title, "characters": len(page.content)})
        return page

    def _refuse(self, reason: str) -> NoReturn:
        self.usage.refused += 1
        raise ToolRefused(reason)

    def _reject(self, reason: str) -> NoReturn:
        self.usage.rejected += 1
        raise ToolRefused(reason)
