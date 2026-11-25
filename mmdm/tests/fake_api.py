"""An offline stand-in for the TypeSafe System One endpoint. No network, no charges."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
from backports import zstd
from typesafe_sdk import AsyncTypeSafeClient

from mmdm.data import Expected, Option, Question, Row
from mmdm.decide import MODEL


Answerer = Callable[[dict[str, Any]], dict[str, Any]]


def leaning_answer(question: dict[str, Any]) -> dict[str, Any]:
    """A valid answer that puts most of the mass on the first option."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.25}
    criteria = question["criteria"]
    keys = [str(i) for i in range(len(criteria))] if question["type"] == "score" else list(criteria)
    rest = 0.3 / (len(keys) - 1)
    probabilities = {k: (0.7 if i == 0 else rest) for i, k in enumerate(keys)}
    if question["type"] == "score":
        legend = {str(i): c["name"] for i, c in enumerate(criteria)}
        return {"type": "score", "score": 0.0, "confidence": 0.7, "legend": legend, "probabilities": probabilities}
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
                "answers": {"decision": answer(payload["questions"]["decision"])},
                "usage": {"input_tokens": 1000, "output_tokens": 3},
            },
        )

    return AsyncTypeSafeClient(api_key="offline-test-key", transport=httpx2.MockTransport(respond))


def row(index: int = 1, *, category: str = "synthetic_category_xyz") -> Row:
    """A labelled row whose category and labels are easy to spot if they leak into a request."""
    return Row(
        id=f"mmdm-{index}",
        input=Question(
            state="The deploy failed after the migration step.",
            question="Which step should the agent retry?",
            options=(
                Option(id="o1", name="Retry migration", criteria="The migration step failed."),
                Option(id="o2", name="Retry build", criteria=None),
                Option(id="o3", name="Retry tests", criteria="Tests failed."),
            ),
            task="choice",
        ),
        expected=Expected(
            label_semantics="categorical_hard",
            probabilities=(1.0, 0.0, 0.0),
            correct_option_ids=("o1",),
            task_category=category,
            task_subcategory="synthetic_subcategory_xyz",
            source="synthetic_source_xyz",
        ),
    )
