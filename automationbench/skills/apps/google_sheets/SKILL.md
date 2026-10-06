---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

## Reading
- Spreadsheet IDs look like ss_*, worksheet IDs like ws_*.
- Always start with google_sheets_get_spreadsheet_by_id(spreadsheet=<ss id>) to list every worksheet {id, title}. Do not guess titles with find_worksheet (it needs the exact title and fails otherwise).
- Then google_sheets_get_many_rows(spreadsheet=<ss id>, worksheet=<ws id>, range "A:Z", row_count 200) for EVERY relevant tab, in parallel. Each row has row_id and cells {Header: value}.
- Tabs named Policy / Rules / Criteria / Overrides / Exclusions / Config / Format / Rate Card / Managers are binding inputs. Tabs marked ARCHIVED or prior-year are not.
- Rows at the bottom with a different schema or statuses like Void/Archived/Superseded/Inactive are noise; ignore them unless the task targets them.
- If you only have a file name, find the ID with google_drive_find_multiple_files (see apps/google_drive).

## Writing
- google_sheets_add_row(spreadsheet, worksheet, cells={Header: value}) with the exact existing headers (case and spacing); fill every column you have data for (including Notes/flags required by policy).
- Before adding, check existing rows for the same key (invoice/PO number, title, email) to avoid duplicates.
- google_sheets_update_row(spreadsheet, worksheet, row=<row_id>, cells={only changed columns}). Update only rows you actually acted on; leave skipped rows untouched.
- Write values in the source's format (keep "$1,234.00" style amounts, dates YYYY-MM-DD) and text flags exactly as policy words them.
