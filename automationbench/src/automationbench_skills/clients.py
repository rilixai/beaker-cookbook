"""Beaker tracing around the native Verifiers HTTP client."""

from __future__ import annotations

import contextvars
from collections.abc import AsyncIterator
from typing import Any, cast

from beaker.tracing import current_trace
from verifiers.v1.clients.client import RelayReply
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects.base import is_sse_done_event
from verifiers.v1.interception.server import InterceptionServer


DEFAULT_MAX_ANSWER_TOKENS = 8192
DEFAULT_MAX_REASONING_TOKENS = 24576


class TracedEvalClient(EvalClient):
    async def relay(self, dialect: Any, body: dict, session_id: str | None = None, headers: Any = None) -> RelayReply:
        # The stream can be consumed/closed in a different asyncio task. Keep
        # the span's context separate so its context tokens close in their owner.
        context = contextvars.copy_context()
        operation = current_trace().model_call(
            operation="chat.completions", provider="openai", model=body["model"], input_messages=body["messages"]
        )
        context.run(operation.__enter__)
        try:
            reply = await super().relay(dialect, body, session_id, headers)
        except BaseException as request_error:
            context.run(operation.__exit__, type(request_error), request_error, request_error.__traceback__)
            raise
        parser = dialect.stream_parser()
        error: BaseException | None = None
        complete = False

        async def chunks() -> AsyncIterator[bytes]:
            nonlocal error, complete
            try:
                async for chunk in reply.chunks:
                    if parser.on_done is not None and is_sse_done_event(chunk):
                        parser.on_done()
                    parser.feed(chunk)
                    yield chunk
                response = parser.finish()
                operation.output(response.raw)
                complete = True
            except BaseException as exc:
                error = exc
                raise

        async def close() -> None:
            try:
                await reply.close()
            finally:
                failure = error or (None if complete else RuntimeError("Model stream did not complete"))
                context.run(operation.__exit__, type(failure) if failure else None, failure, None)

        return RelayReply(content_type=reply.content_type, chunks=chunks(), close=close)


class TracedInterception(InterceptionServer):
    """Use native transport and retries with run-scoped credentials and tracing."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(requires_tunnel=False)
        self.api_key = api_key

    def _client(self, config: Any) -> TracedEvalClient:
        key = config.model_dump_json()
        if key not in self.clients:
            client = TracedEvalClient(config)
            if self.api_key is not None:
                client.api_key = self.api_key
            self.clients[key] = client
            self.stack.push_async_callback(client.close)
        return cast(TracedEvalClient, self.clients[key])
