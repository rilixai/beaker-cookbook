---
name: google_sheets
include_in_listing: true
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

This is the operating procedure for the Google Sheets app (rows, lookups, worksheets, updates, etc.).

Before using a workbook, call `google_sheets_get_spreadsheet_by_id` with its spreadsheet ID to list worksheet IDs, titles, and headers. Reuse metadata already read for that workbook.

Read the rows in the task's data worksheets, plus sibling worksheets that govern the workflow: policies, overrides, eligibility lists, routing, pricing, or rates. Metadata lists tabs; it does not supply their row values. Apply the relevant rules and updates before choosing records, calculating values, or writing.

Use the returned worksheet IDs for row reads and writes. The workbook title is not a worksheet title. `google_sheets_find_worksheet` requires an exact, case-sensitive title; a failed guess does not show that a worksheet is absent. If a lookup fails or a guessed tab returns no rows, inspect the workbook metadata and retry with the actual worksheet ID.

Before creating a replacement worksheet, check the metadata for the existing destination. Use the existing worksheet when present; create a new one only when the task needs it and it is absent.
