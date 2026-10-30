---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

This is the operating procedure for the Google Sheets app (rows, lookups, worksheets, updates, etc.).

To discover a spreadsheet's worksheets, call `google_sheets_get_spreadsheet_by_id` with its ID. The response lists worksheet IDs, titles, and headers; `includeGridData=true` does not read their rows. Use `google_sheets_get_many_rows` with the returned worksheet IDs to read the relevant contents. Use this list instead of guessing worksheet names. When the spreadsheet supports a business workflow, inspect its related rules, configuration, exclusions, and overrides alongside the task data before selecting records or writing changes. Follow references to other policy sources.
