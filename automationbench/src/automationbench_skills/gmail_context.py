"""Expose related message headings alongside the first filtered Gmail search."""

from __future__ import annotations

import json
from functools import wraps
from typing import Callable
from weakref import WeakValueDictionary

MAX_INDEX_BYTES = 16_000
MESSAGE_LIMIT = 100
SEARCH_TOOLS = {"gmail_find_email", "gmail_list_emails"}


def _size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _mailbox_index(execute: Callable, world, returned_ids: set) -> dict:
    listing = json.loads(execute(
        world=world,
        tool_name="gmail_find_email",
        arguments=json.dumps({"query": "", "format": "metadata", "max_results": MESSAGE_LIMIT}),
    ))
    if not listing.get("success"):
        return {"error": "Mailbox index unavailable"}
    index = {
        "note": "Other mailbox message headings, supplied once per rollout. These may fall outside the search query. Scan for rules, corrections, and updates relevant to the task; read those messages in full by ID before using them. Compare authority, scope, and explicit updates. Ignore unrelated messages. If truncated, widen the search to find more.",
        "messages": [],
        "truncated": bool(listing.get("has_more")),
        "message_limit": MESSAGE_LIMIT,
    }
    for message in listing.get("messages", []):
        if message.get("id") in returned_ids:
            continue
        if {str(label).upper() for label in message.get("label_ids", [])} & {"SPAM", "TRASH"}:
            continue
        heading = {key: message[key] for key in ("id", "from", "subject", "date", "label_ids") if key in message}
        index["messages"].append(heading)
        if _size(index) > MAX_INDEX_BYTES:
            index["messages"].pop()
            index["truncated"] = True
    return index


def with_gmail_context(execute: Callable) -> Callable:
    """Preserve dispatch and matches; add a bounded heading index once per world."""
    indexed = WeakValueDictionary()

    @wraps(execute)
    def execute_with_index(world, tool_name: str, arguments: str) -> str:
        result = execute(world=world, tool_name=tool_name, arguments=arguments)
        if tool_name not in SEARCH_TOOLS or id(world) in indexed:
            return result
        try:
            args = json.loads(arguments)
            query = args.get("query") or ""
            if args.get("id") or (query.strip() in {"", "*"} and not args.get("label")):
                return result
            response = json.loads(result)
            if response.get("success") is not True or not isinstance(response.get("messages"), list):
                return result
            indexed[id(world)] = world
            returned_ids = {message.get("id") for message in response["messages"]}
            index = _mailbox_index(execute, world, returned_ids)
            if not index.get("messages") and not index.get("truncated") and not index.get("error"):
                return result
            response["mailbox_index"] = index
        except Exception:
            # An optional discovery read must not break a successful search.
            return result
        return json.dumps(response, ensure_ascii=False)

    return execute_with_index
