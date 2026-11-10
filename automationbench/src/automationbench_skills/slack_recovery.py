"""Supply bounded channel context when a literal Slack search misses."""

from __future__ import annotations

import json
from functools import wraps
from typing import Callable
from weakref import WeakValueDictionary

MAX_CONTEXT_BYTES = 16_000
MAX_CHANNELS = 10
HISTORY_LIMIT = 100
SEARCH_TOOLS = {"slack_find_message", "slack_find_message_in_channel"}


def _encoded_size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _channel_context(execute: Callable, world) -> dict:
    """Read through the existing app tools; never inspect private world fields."""
    listing = json.loads(execute(world=world, tool_name="slack_list_channels", arguments="{}"))
    if not listing.get("success"):
        return {"error": str(listing.get("error", "Channel listing failed"))[:500]}
    channels = listing.get("channels", [])
    context = {
        "note": "The query had no literal match. These are recent channel messages, not search matches. Use only guidance relevant to this task, compare authority and scope, and follow references. Read channels or threads explicitly if more history is needed. This fallback is supplied once per rollout.",
        "channels": [],
        "truncated": len(channels) > MAX_CHANNELS,
        "history_limit_per_channel": HISTORY_LIMIT,
    }
    histories = []
    for channel in channels[:MAX_CHANNELS]:
        item = {"id": channel["id"], "name": channel["name"], "messages": []}
        context["channels"].append(item)
        if _encoded_size(context) > MAX_CONTEXT_BYTES:
            context["channels"].pop()
            context["truncated"] = True
            continue
        history = json.loads(execute(
            world=world,
            tool_name="slack_list_channel_messages",
            arguments=json.dumps({"channel": channel["id"], "limit": HISTORY_LIMIT}),
        ))
        if not history.get("success"):
            context["truncated"] = True
            continue
        messages = history.get("messages", [])
        if len(messages) >= HISTORY_LIMIT:
            context["truncated"] = True
        histories.append((item, messages))
    # Share the output budget across channels so a busy channel does not hide
    # a correction or a referenced policy in another channel.
    for offset in range(HISTORY_LIMIT):
        for item, messages in histories:
            if offset >= len(messages):
                continue
            message = {k: messages[offset][k] for k in ("ts", "user", "text") if k in messages[offset]}
            item["messages"].append(message)
            if _encoded_size(context) > MAX_CONTEXT_BYTES:
                item["messages"].pop()
                context["truncated"] = True
    return context


def with_slack_recovery(execute: Callable) -> Callable:
    """Keep dispatch and successful replies intact; augment the first miss only."""
    recovered = WeakValueDictionary()

    @wraps(execute)
    def execute_with_context(world, tool_name: str, arguments: str) -> str:
        result = execute(world=world, tool_name=tool_name, arguments=arguments)
        if tool_name not in SEARCH_TOOLS or id(world) in recovered:
            return result
        response = json.loads(result)
        if response.get("success") is not False or not str(response.get("error", "")).startswith("No messages found matching "):
            return result
        recovered[id(world)] = world
        try:
            context = _channel_context(execute, world)
            if not context.get("error") and not any(ch["messages"] for ch in context.get("channels", [])):
                return result
            response["channel_context"] = context
        except Exception as exc:
            # Recovery must not turn a normal empty search into a failed tool.
            response["channel_context"] = {"error": f"Channel context unavailable: {exc}"[:500]}
        return json.dumps(response, ensure_ascii=False)

    return execute_with_context
