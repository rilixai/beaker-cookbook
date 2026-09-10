---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

# Google Sheets

## Reading: read every worksheet, every row, every column
1. Locate the spreadsheet (`google_drive_find_multiple_files` with the title, or the ID
   if known) and call `google_sheets_get_spreadsheet_by_id` to list ALL worksheets.
2. `google_sheets_get_many_rows(spreadsheet, worksheet, range: "A:Z", row_count: 200)`
   on every worksheet - not just the obvious data tab. Policy tabs ("Billing Policy",
   "Blocked Vendors", "Rate Card", "Categories") contain rules that change the answer.
   Worksheets marked ARCHIVED / prior-year are reference only - never take values from
   them.
3. Read the `Notes` column of every data row. Notes carry overrides: rate overrides,
   payment plans, duplicate flags, reclassifications, clarifications about similarly
   named entities. Apply them.
4. Rows with a different column schema (e.g. vendor payment rows mixed into an expense
   log) or dates outside the requested range are not part of the task set.

## Filtering and arithmetic
- Filter on the exact status/category values the policy names (e.g. Status = `Approved`
  only; exclude `Pending Approval`, `Rejected`, `Personal`).
- Recompute totals from the filtered rows after applying overrides, corrections and
  discounts. Write the arithmetic out before entering it.

## Writing
- `google_sheets_add_row(spreadsheet, worksheet, cells: {header: value})` - the keys
  must match the worksheet headers exactly (case and spacing). Fill every header column
  the source provides (vendor, invoice number, date, amount, due date, notes...).
- Copy identifiers and amounts verbatim from the source (or from the posted correction),
  e.g. `GL-88210`, `$11,340.50`, `PO-2026-0155`. Do not round or reformat.
- Put required flag text in the designated column exactly as the policy words it
  (e.g. Notes = `REVIEW - over threshold`).
- `google_sheets_update_row(spreadsheet, worksheet, row: <row_id>, cells: {...})` - update only the
  rows you acted on. Never touch rows that the process says to skip (payment plans,
  excluded entities); an untouched skipped row is part of the expected result.
- Check for duplicates before adding (same invoice/PO number already present -> skip and
  give the required notice instead).
- After writing, re-read the worksheet and confirm each added/updated row's values.
