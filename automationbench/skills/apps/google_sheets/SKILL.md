---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

# Google Sheets

## Reading
1. Locate the spreadsheet (`google_drive_find_multiple_files` by title, or the ID if
   known), then `google_sheets_get_spreadsheet_by_id` to list ALL worksheets.
2. `google_sheets_get_many_rows(spreadsheet, worksheet, range="A:Z", row_count=200)` on
   every worksheet, not just the obvious data tab - policy/reference tabs change the
   answer. Tabs marked archived or prior-year are reference only; never take values
   from them.
3. Read the Notes column of every data row; it carries per-row overrides,
   exclusions and clarifications. Apply them.
4. Rows with a different column schema or dates outside the requested range are not part
   of the task set.

## Filtering and arithmetic
- Filter on the exact status/category values the policy names.
- Recompute totals from the filtered rows after applying overrides, corrections and
  discounts; write the arithmetic out first.

## Writing
- `google_sheets_add_row(spreadsheet, worksheet, cells={header: value})`; keys must
  match the worksheet headers exactly. Fill every header column the source provides.
- Copy identifiers and amounts verbatim from the source (or the posted correction); do
  not round or reformat. Put required flag text in the designated column exactly as the
  policy words it.
- `google_sheets_update_row(spreadsheet, worksheet, row=<row_id>, cells={...})` only for
  rows you acted on. Never touch rows the process says to skip.
- Check for duplicates before adding; if the identifier already exists, skip and give the
  required notice instead.
- After writing, re-read the worksheet and confirm each added/updated row.
