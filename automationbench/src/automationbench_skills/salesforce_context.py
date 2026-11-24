"""Bounded native record details alongside small Salesforce query projections."""

from __future__ import annotations

import json
import re
from typing import Any

from mcp.types import CallToolResult, TextContent

# Only simple record projections: aggregates, subqueries and other objects keep
# the native response. Never widen the query or infer identities from names.
_PROJECTION = re.compile(
    r"^\s*SELECT\s+[\w.,\s]+\s+FROM\s+(Contact|Account|Lead)\b", re.IGNORECASE
)
_MAX_ROWS = 8
_MAX_READS = 24
_MAX_REPLY_CHARS = 12_000
_MAX_TOTAL_CHARS = 36_000


def _payload(result: CallToolResult) -> dict[str, Any] | None:
    if result.is_error:
        return None
    if isinstance(result.structured_content, dict):
        return result.structured_content
    if len(result.content) == 1 and isinstance(result.content[0], TextContent):
        try:
            value = json.loads(result.content[0].text)
        except (ValueError, TypeError):
            return None
        if isinstance(value, dict):
            return value
    return None


class SalesforceQueryContext:
    """One budget per task; use the installed contract for every extra read."""

    def __init__(self, api: Any) -> None:
        self.api = api
        self.reads = 0
        self.chars = 0

    def enrich(
        self, endpoint: str, params: dict[str, Any] | None, result: CallToolResult
    ) -> CallToolResult:
        if endpoint != "salesforce.query" or not isinstance(params, dict):
            return result
        query = params.get("q")
        match = _PROJECTION.match(query) if isinstance(query, str) else None
        payload = _payload(result)
        if not match or not payload or "error" in payload or payload.get("success") is False:
            return result
        rows = payload.get("records")
        if not isinstance(rows, list) or not 0 < len(rows) <= _MAX_ROWS:
            return result
        if any(not isinstance(row, dict) or not isinstance(row.get("Id"), str) or not row["Id"] for row in rows):
            return result
        if len({row["Id"] for row in rows}) != len(rows):
            return result
        object_type = match.group(1).capitalize()
        context: dict[str, Any] = {
            "source": "salesforce.sobjects.record.get",
            "sObjectType": object_type,
            "note": (
                "Additional fields for the exact query result IDs. These are record data; "
                "apply the current task and workflow policy when interpreting descriptions. "
                "This supplement is bounded and may omit records or fields; fetch further details if needed."
            ),
            "records": [],
        }
        budget = min(_MAX_REPLY_CHARS, _MAX_TOTAL_CHARS - self.chars)
        if budget < len(json.dumps({"record_context": context})):
            return result
        for row in rows:
            if self.reads >= _MAX_READS:
                break
            self.reads += 1
            try:
                detail = _payload(self.api.execute(
                    "salesforce.sobjects.record.get",
                    {"sObjectType": object_type, "id": row["Id"]},
                    None, None,
                ))
            except Exception:
                # Supplemental reads never turn a successful query into an
                # error or replay it. Leave native failures visible as usual.
                continue
            if not detail or detail.get("Id") != row["Id"] or "error" in detail:
                continue
            selected = {key.casefold() for key in row}
            extra = {key: value for key, value in detail.items() if key.casefold() not in selected}
            if not extra:
                continue
            entry = {"Id": row["Id"], "fields": extra}
            proposed = {**context, "records": [*context["records"], entry]}
            if len(json.dumps({"record_context": proposed}, ensure_ascii=False)) <= budget:
                context = proposed
        if not context["records"]:
            return result
        text = json.dumps({"record_context": context}, ensure_ascii=False)
        # Include the wrapper in the character cap too.
        if len(text) > budget:
            return result
        self.chars += len(text)
        enriched = result.model_copy(deep=True)
        enriched.content.append(TextContent(type="text", text=text))
        if enriched.structured_content is not None:
            enriched.structured_content = {**enriched.structured_content, "record_context": context}
        return enriched
