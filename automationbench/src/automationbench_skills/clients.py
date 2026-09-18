from __future__ import annotations

from typing import Any

from automationbench.clients import RetryingOpenAIChatCompletionsClient, _perf


class CostTrackingChatCompletionsClient(RetryingOpenAIChatCompletionsClient):
    async def get_native_response(self, *args: Any, **kwargs: Any) -> Any:
        resp = await super().get_native_response(*args, **kwargs)
        record_cost(kwargs.get("state"), resp)
        return resp


def record_cost(state: Any, native_response: Any) -> None:
    usage = getattr(native_response, "usage", None)
    if usage is None:
        return
    if isinstance(usage, dict):
        cost = usage.get("cost")
    else:
        cost = getattr(usage, "cost", None)
        if cost is None:
            model_extra = getattr(usage, "model_extra", None) or {}
            cost = model_extra.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return
    perf = _perf(state)
    if perf is not None:
        perf["cost_usd"] = perf.get("cost_usd", 0.0) + float(cost)
