"""Run Toolathlon's migrated main entrypoint with the native OTLP connector."""

from __future__ import annotations

import asyncio

from main import main
from toolathlon_native_trace import export_native_trace


async def run() -> int:
    async with export_native_trace():
        return await main()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
