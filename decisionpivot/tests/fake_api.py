"""An offline stand-in for the TypeSafe System One endpoint, and small labelled rows. No network, no charges."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
from backports import zstd
from typesafe_sdk import AsyncTypeSafeClient

from decisionpivot.data import CaseInput, Expected, Item, Question, Row
from decisionpivot.decide import MODEL


Answerer = Callable[[dict[str, Any], Any], dict[str, Any]]


def leaning_answer(question: dict[str, Any], state: Any) -> dict[str, Any]:
    """A valid answer that puts most of the mass on the first option."""
    del state
    keys = list(question["criteria"])
    rest = 0.3 / (len(keys) - 1)
    probabilities = {k: (0.7 if i == 0 else rest) for i, k in enumerate(keys)}
    return {"type": "choice", "choice": keys[0], "confidence": 0.7, "probabilities": probabilities}


def client(
    received: list[dict[str, Any]] | None = None,
    *,
    answer: Answerer = leaning_answer,
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
        return httpx2.Response(
            200,
            json={
                "model": model,
                "answers": {"q": answer(payload["questions"]["q"], payload["state"])},
                "usage": {"input_tokens": 1000, "output_tokens": 3},
            },
        )

    return AsyncTypeSafeClient(api_key="offline-test-key", transport=httpx2.MockTransport(respond))


QUESTION = Question(
    type="choice",
    instructions="Can the booking be confirmed?",
    criteria={"yes": "The booking can be confirmed.", "no": "The booking cannot be confirmed."},
)


def row(index: int = 1, *, golds: tuple[str, ...] = ("yes", "no"), family: str = "sched_conflict") -> Row:
    """A contrast group whose members differ in one planted detail (the room's seat count)."""
    items = tuple(
        Item(
            id=f"{family}:{index:03d}{chr(97 + n)}",
            state=f"Room Kestrel seats {10 + n * 4} people. The booking is for 12 people at 09:00 on 3 March.",
            questions={"q": QUESTION},
        )
        for n in range(len(golds))
    )
    return Row(
        id=f"dp-{family}:g{index:04d}",
        input=CaseInput(family=family, items=items),
        expected=Expected(gold={i.id: g for i, g in zip(items, golds, strict=True)}, level=3, layer="D"),
    )
