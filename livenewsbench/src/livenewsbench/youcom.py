"""You.com: the Inference Router for the model, the Search and Contents APIs for the web.

API references (checked 2026-11-12):
- https://you.com/docs/api-reference/inference-router/create-chat-completion
- https://you.com/docs/api-reference/search/v1-search
- https://you.com/docs/api-reference/contents
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx
from openai import AsyncOpenAI

from livenewsbench.chat import TransientServiceError
from livenewsbench.tools import Page, SearchExtraction, SearchHit


API_KEY_ENV = "YDC_API_KEY"
ROUTER_BASE_URL = "https://api.you.com/v1"
ROUTER_HOST = "api.you.com"
SEARCH_URL = "https://ydc-index.io/v1/search"
CONTENTS_URL = "https://ydc-index.io/v1/contents"
DEFAULT_MODEL = "deepseek-v4.1-flash"
# Reasoning tokens count against the output limit, so leave room for them. Other
# sampling settings are omitted: the router then applies each vendor's reference values.
REQUEST_OPTIONS: dict[str, Any] = {"max_tokens": 16384}


@dataclass(frozen=True)
class TokenPrice:
    input: float
    cached_input: float
    output: float


# USD per 1M tokens, https://you.com/product/inference-router/models (checked 2026-11-12).
ROUTER_PRICES = {
    "deepseek-v4.1-flash": TokenPrice(0.30, 0.007, 1.20),
    "deepseek-v4-pro": TokenPrice(1.65, 0.10, 3.65),
    "glm-5.3": TokenPrice(1.35, 0.26, 4.25),
    "glm-5.3-flash": TokenPrice(0.145, 0.029, 0.484),
    "qwen3.8-flash": TokenPrice(0.139, 0.015, 0.435),
    "qwen3.8-max": TokenPrice(1.85, 0.231, 5.55),
    "kimi-k3": TokenPrice(2.85, 0.285, 14.25),
    "kimi-k3.1-pro": TokenPrice(1.90, 0.19, 9.50),
    "minimax-m3": TokenPrice(0.291, 0.058, 1.163),
    "gpt-oss-120b": TokenPrice(0.097, 0.053, 0.727),
}
# USD per call, https://you.com/docs/administration/billing (checked 2026-11-12).
SEARCH_PRICE = 5.00 / 1000
PAGE_PRICE = 1.00 / 1000

RETRY_STATUSES = {429, 500, 502, 503, 504}
ATTEMPTS = 6


def api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} to a You.com API key.")
    return key


def router_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=ROUTER_BASE_URL, api_key=key, max_retries=5, timeout=300)


class YouComWeb:
    """Search API results and Contents API pages, retried on rate limits and server errors."""

    def __init__(self, key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(headers={"X-API-Key": key}, timeout=60, transport=transport)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._client.aclose()

    async def search(
        self, query: str, *, count: int, extraction: SearchExtraction, exclude_domains: tuple[str, ...]
    ) -> list[SearchHit]:
        body: dict[str, Any] = {"query": query, "count": count, "exclude_domains": list(exclude_domains)}
        if extraction == "highlights":
            body["extraction"] = {"extraction_mode": "highlights"}
        payload = await self._post(SEARCH_URL, body)
        hits = []
        for result in payload.get("results", {}).get("web", []) or []:
            contents = result.get("contents") or {}
            passages = contents.get("highlights") or result.get("snippets") or [result.get("description") or ""]
            hits.append(
                SearchHit(
                    url=result["url"],
                    title=result.get("title") or "",
                    snippet="\n".join(str(passage) for passage in passages if passage),
                    published=result.get("page_age"),
                )
            )
        return hits

    async def fetch(self, url: str) -> Page:
        payload = await self._post(CONTENTS_URL, {"urls": [url], "formats": ["markdown"]})
        pages = payload if isinstance(payload, list) else []
        if not pages:
            return Page(url=url, title="", content="The page could not be fetched.")
        page = pages[0]
        return Page(url=page.get("url") or url, title=page.get("title") or "", content=page.get("markdown") or "")

    async def _post(self, url: str, body: dict[str, Any]) -> Any:
        for attempt in range(ATTEMPTS):
            try:
                response = await self._client.post(url, json=body)
            except httpx.TransportError as exc:
                error: str = type(exc).__name__
            else:
                if response.status_code not in RETRY_STATUSES:
                    if response.status_code >= 400:
                        # Keep the body out: it can echo the request. The status says enough.
                        raise RuntimeError(f"You.com {url.rsplit('/', 1)[-1]} failed with HTTP {response.status_code}")
                    return response.json()
                error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("retry-after", "")
                if retry_after.isdigit() and attempt < ATTEMPTS - 1:
                    await asyncio.sleep(min(float(retry_after), 30.0))
                    continue
            if attempt < ATTEMPTS - 1:
                await asyncio.sleep(min(2.0**attempt, 20.0) * (0.5 + random.random()))
        raise TransientServiceError(f"You.com {url.rsplit('/', 1)[-1]} failed after {ATTEMPTS} attempts: {error}")
