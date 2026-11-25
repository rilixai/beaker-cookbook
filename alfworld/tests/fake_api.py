"""Offline stand-ins: the TypeSafe System One endpoint and one real game. No paid calls."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx2
from backports import zstd
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from alfworld_agent.data import Row, games_root, make_rows
from alfworld_agent.decide import MODEL


# The smallest valid_seen game; its walkthrough wins in four steps. Fetched by games_root, never committed.
GAME = "valid_seen/pick_and_place_simple-SprayBottle-None-GarbageCan-409/trial_T20190908_054742_818364/game.tw-pddl"

# Picks a criteria key given the request payload.
Chooser = Callable[[dict[str, Any]], str]


def first_option(payload: dict[str, Any]) -> str:
    return next(iter(payload["questions"]["decision"]["criteria"]))


def scripted(actions: list[str]) -> Chooser:
    """Replays a known action list, one per request."""
    remaining = iter(actions)
    return lambda payload: next(remaining)


def client(
    received: list[dict[str, Any]] | None = None,
    *,
    choose: Chooser = first_option,
    model: str = MODEL,
    status: int = 200,
) -> AsyncTypeSafeClient:
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/systemone"
        body = request.content
        if request.headers.get("content-encoding") == "zstd":
            body = zstd.decompress(body)
        payload = json.loads(body)
        if received is not None:
            received.append(payload)
        if status != 200:
            return httpx2.Response(status, json={"error": {"message": "offline failure"}})
        keys = list(payload["questions"]["decision"]["criteria"])
        choice = choose(payload)
        rest = 0.2 / (len(keys) - 1)
        probabilities = {k: (0.8 if k == choice else rest) for k in keys}
        answer = {"type": "choice", "choice": choice, "confidence": 0.8, "probabilities": probabilities}
        return httpx2.Response(
            200,
            json={"model": model, "answers": {"decision": answer}, "usage": {"input_tokens": 500, "output_tokens": 2}},
        )

    return AsyncTypeSafeClient(
        api_key="offline-test-key", transport=httpx2.MockTransport(respond), retry=RetryPolicy(max_retries=0)
    )


def game_path() -> Path:
    return games_root() / GAME


def walkthrough() -> list[str]:
    actions: list[str] = json.loads(game_path().read_text())["walkthrough"]
    return actions


def row() -> Row:
    [only] = make_rows("valid_seen", [GAME])
    return only


@contextmanager
def patched_client(monkeypatch: Any, factory: Callable[[], AsyncTypeSafeClient]) -> Iterator[list[dict[str, Any]]]:
    from alfworld_agent import decide

    built: list[dict[str, Any]] = []

    def make(**kwargs: Any) -> AsyncTypeSafeClient:
        built.append(kwargs)
        return factory()

    monkeypatch.setattr(decide, "AsyncTypeSafeClient", make)
    yield built
