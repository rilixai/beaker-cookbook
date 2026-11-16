"""Scripted Chat Completions endpoint for native-runtime tests (no model charges)."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web


class ScriptedEndpoint:
    def __init__(self, turns: list[dict]) -> None:
        self.turns = turns
        self.requests: list[dict] = []
        self.authorizations: list[str | None] = []

    async def __aenter__(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.respond)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
        return self

    async def __aexit__(self, *args):
        await self.runner.cleanup()

    async def respond(self, request):
        body = await request.json()
        index = len(self.requests)
        self.requests.append(body)
        self.authorizations.append(request.headers.get("Authorization"))
        turn = self.turns[index] if index < len(self.turns) else {"content": "Done"}
        if "status" in turn:
            return web.json_response({"error": {"message": "scripted provider error"}}, status=turn["status"])
        await asyncio.sleep(turn.get("delay", 0))
        message = {"role": "assistant", "content": turn.get("content")}
        if calls := turn.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "index": n,
                    "id": f"call-{index}-{n}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                }
                for n, call in enumerate(calls)
            ]
        common = {"id": f"reply-{index}", "object": "chat.completion.chunk", "created": 1, "model": body["model"]}
        chunks = [
            {**common, "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
            {
                **common,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        return web.Response(
            text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )
