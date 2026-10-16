---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

When appending rows to an existing worksheet, use its actual column headers. An empty row read does not mean the worksheet has no columns. If the headers are not already known, discover and call `google_sheets_get_spreadsheet_by_id`, then select the destination worksheet by its ID and read its `headers`. Do not insert header or sample rows just to discover the columns.

Build `cells` with those destination headers as keys. Fill each column for which the source record or workflow supplies a value, preserving the source text. Do not invent missing values. Reuse headers already returned by metadata or existing rows rather than repeating the lookup.
