---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

This is the operating procedure for the Google Sheets app (rows, lookups, worksheets, updates, etc.).

- Open every worksheet of a spreadsheet (get_spreadsheet_by_id lists them) and read each with get_many_rows; policy/config/updates/blackout tabs override the base rules.
- Write status/label cells using the exact vocabulary in the rules or existing rows.
