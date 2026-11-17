"""Controlled wrapper faults for data-quality evaluation cases.

These change API observations or forwarded arguments, never task data or graders.
"""

from __future__ import annotations

import json
from typing import Any, cast

from automationbench.tools.api.contract import APIContract
from mcp.types import CallToolResult, TextContent


class ScenarioAPIContract(APIContract):
    def __init__(self, world: Any, fault: str | None) -> None:
        super().__init__(world)
        self.fault = fault

    def execute(
        self,
        endpoint_id: str,
        path: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> CallToolResult:
        opportunity_update = endpoint_id == "salesforce.sobjects.opportunity.update" or (
            endpoint_id == "salesforce.sobjects.record.update" and (path or {}).get("sObjectType") == "Opportunity"
        )
        if self.fault == "drop_opportunity_amount" and opportunity_update and isinstance(body, dict):
            body = {key: value for key, value in body.items() if key not in {"Amount", "amount"}}
        result = super().execute(endpoint_id, path, params, body)
        if self.fault == "empty_email_lookup" and endpoint_id == "gmail.users.messages.list" and not result.is_error:
            # Preserve the API result envelope and the authoritative emails in the world.
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"messages": [], "resultSizeEstimate": 0}))]
            )
        return cast(CallToolResult, result)
