"""Load a candidate agent, run one question, and meter what it used."""

from __future__ import annotations

import importlib.util
import sys
import time
from collections.abc import Awaitable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from livenewsbench.chat import CallLimitReached, ChatModel, ChatUsage
from livenewsbench.tools import WebTools, WebUsage
from livenewsbench.youcom import PAGE_PRICE, ROUTER_PRICES, SEARCH_PRICE


MAX_MODEL_CALLS = 12
MODEL_ENV = "LIVENEWSBENCH_MODEL"
SEED_AGENT_DIR = Path(__file__).resolve().parent / "agent"


class Agent(Protocol):
    def __call__(self, question: str, *, chat: ChatModel, web: WebTools) -> Awaitable[str]: ...


_loaded: dict[Path, Agent] = {}


def load_agent(agent_dir: Path) -> Agent:
    """Import an agent package from a directory, so a candidate's copy wins over the installed seed."""
    agent_dir = agent_dir.resolve()
    if agent_dir not in _loaded:
        name = f"livenewsbench_agent_{len(_loaded)}"
        spec = importlib.util.spec_from_file_location(
            name, agent_dir / "__init__.py", submodule_search_locations=[str(agent_dir)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"No agent package in {agent_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _loaded[agent_dir] = cast(Agent, module.answer_question)
    return _loaded[agent_dir]


@dataclass
class Rollout:
    answer: str
    finish: str
    seconds: float
    chat: ChatUsage
    web: WebUsage

    def usage(self) -> dict[str, Any]:
        return {"finish": self.finish, "seconds": self.seconds, "model": asdict(self.chat), "web": asdict(self.web)}


async def run_rollout(question: str, *, agent: Agent, chat: ChatModel, web: WebTools) -> Rollout:
    started = time.perf_counter()
    try:
        answer, finish = await agent(question, chat=chat, web=web), "answered"
    except CallLimitReached:
        answer, finish = "", "call_limit"
    return Rollout(answer.strip(), finish, time.perf_counter() - started, chat.usage, web.usage)


def list_price_usd(model: str | None, chat: ChatUsage, web: WebUsage) -> dict[str, float | None]:
    """You.com list prices applied to measured usage; the model share is None off the router."""
    price = ROUTER_PRICES.get(model) if model else None
    model_usd = (
        None
        if price is None
        else (
            (chat.input_tokens - chat.cached_input_tokens) * price.input
            + chat.cached_input_tokens * price.cached_input
            + chat.output_tokens * price.output
        )
        / 1e6
    )
    search_usd = web.searches * SEARCH_PRICE
    pages_usd = web.page_visits * PAGE_PRICE
    total = None if model_usd is None else model_usd + search_usd + pages_usd
    return {"model": model_usd, "search": search_usd, "pages": pages_usd, "total": total}
