"""Resolve named table updates to sparse native Sheets writes."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any, get_type_hints

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field


class RowUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_value: str = Field(description="Exact displayed cell value in the key column; never a row number.")
    values: dict[str, str | int | float | bool] = Field(
        description="Exact existing column headers mapped to the requested new cell values."
    )


def _reply(payload: dict[str, Any], *, error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=error,
    )


def _payload(result: CallToolResult) -> dict[str, Any]:
    if result.structured_content is not None:
        return result.structured_content
    for part in result.content:
        if isinstance(part, TextContent):
            value = json.loads(part.text)
            if isinstance(value, dict):
                return value
    raise ValueError("Sheets did not return a JSON object")


def _column(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def make_sheets_update_rows(fetch: Callable[..., CallToolResult]) -> Callable[..., CallToolResult]:
    def sheets_update_rows(
        spreadsheet_id: str,
        worksheet: str,
        key_column: str,
        updates: list[RowUpdate],
    ) -> CallToolResult:
        """Update existing table rows by exact record keys and named columns.

        Prefer this for status/field changes in an existing table instead of calculating A1
        coordinates or rebuilding rows. Supply the worksheet title or ID, a unique key column
        header, and updates with key_value and values (header: new value). Read the business
        data and decide eligibility/values first. Only include records and fields to change.
        This reads the live table, resolves every key/header before writing, and preserves
        other cells and headers. Missing/ambiguous keys or headers cause no writes.
        Updates use RAW values. Use api_fetch for appending rows, creating sheets, formulas,
        formatting, or tables without a unique key column or headers in row 1. If a native write fails, earlier
        writes remain; the error lists completed cells. Do not replay a successful batch.
        """
        if not updates or not worksheet or not key_column:
            return _reply({"error": "Provide a worksheet, key column, and at least one update."}, error=True)
        quoted = "'" + worksheet.replace("'", "''") + "'"
        # GridData gives absolute row offsets, all columns, and blank rows without
        # relying on a bounded values.get result or inferring positions from keys.
        read = fetch(
            endpoint="sheets.spreadsheets.get",
            path={"spreadsheetId": spreadsheet_id},
            params={"ranges": [quoted], "includeGridData": True},
        )
        if read.is_error:
            return read
        try:
            document = _payload(read)
            sheets = [sheet for sheet in document.get("sheets", []) if sheet.get("data")]
            if len(sheets) != 1 or len(sheets[0]["data"]) != 1:
                raise ValueError("The worksheet must resolve to one complete grid.")
            sheet = sheets[0]
            grid = sheet["data"][0]
            if grid.get("startRow", 0) != 0 or grid.get("startColumn", 0) != 0:
                raise ValueError("The table must start at A1 with headers in row 1.")
            rows = grid.get("rowData", [])
            headers: dict[str, list[int]] = defaultdict(list)
            for column, cell in enumerate(rows[0].get("values", []), start=1):
                label = cell.get("formattedValue", "")
                if label:
                    headers[label].append(column)
            if len(headers.get(key_column, [])) != 1:
                raise ValueError(f"Key header {key_column!r} is missing or ambiguous.")
            key_index = headers[key_column][0] - 1
            matches: dict[str, list[int]] = defaultdict(list)
            for row_number, row in enumerate(rows[1:], start=2):
                cells = row.get("values", [])
                if key_index < len(cells) and "formattedValue" in cells[key_index]:
                    matches[cells[key_index]["formattedValue"]].append(row_number)
            title = sheet["properties"]["title"]
            prefix = "'" + title.replace("'", "''") + "'!"
            plan: dict[tuple[int, int], Any] = {}
            for update in updates:
                if not update.key_value or len(matches.get(update.key_value, [])) != 1:
                    raise ValueError(f"Record key {update.key_value!r} is missing or ambiguous; no writes made.")
                if not update.values:
                    raise ValueError("Each record must specify at least one named field.")
                row_number = matches[update.key_value][0]
                for label, value in update.values.items():
                    if len(headers.get(label, [])) != 1:
                        raise ValueError(f"Target header {label!r} is missing or ambiguous; no writes made.")
                    coordinate = row_number, headers[label][0]
                    if coordinate in plan:
                        raise ValueError("A cell was targeted more than once; no writes made.")
                    plan[coordinate] = value
        except (ValueError, KeyError, IndexError, TypeError) as error:
            return _reply({"error": str(error)}, error=True)

        completed = []
        # Native values.batchUpdate is not available in this API catalog. Keep
        # each sparse write on the normal execution/state-export path instead.
        for (row_number, column), value in plan.items():
            cell_range = f"{prefix}{_column(column)}{row_number}"
            result = fetch(
                endpoint="sheets.spreadsheets.values.update",
                path={"spreadsheetId": spreadsheet_id, "range": cell_range},
                params={"valueInputOption": "RAW"},
                body={"majorDimension": "ROWS", "values": [[value]]},
            )
            if result.is_error:
                return _reply({
                    "error": "Native update failed; earlier completed writes remain.",
                    "failed_range": cell_range,
                    "native_result": result.model_dump(mode="json"),
                    "completed": completed,
                }, error=True)
            completed.append({"range": cell_range, "value": value})
        return _reply({"spreadsheetId": spreadsheet_id, "updatedCells": len(completed), "completed": completed})

    # The native state/serialization decorators live in another module. Resolve
    # RowUpdate here so MCP can build its schema after those wrappers are added.
    sheets_update_rows.__annotations__ = get_type_hints(sheets_update_rows)
    return sheets_update_rows
