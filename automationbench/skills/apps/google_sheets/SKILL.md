---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

For changes to existing table rows, prefer `sheets_update_rows` when available. Read the source records and applicable rules first. Choose an existing column that uniquely identifies each record. Supply its exact header as `key_column`, then each record's exact displayed `key_value` and a `values` object mapping existing column headers to the requested new values. Include only records and fields that should change. The tool resolves live row and column positions; do not substitute a filtered-list index for the record key.

If a key or header is missing or ambiguous, inspect the source and resolve the match before retrying. Do not create or rebuild a table to get around a failed lookup. A later native write error can leave earlier cells changed; read the reported completed ranges before continuing.

Use `api_fetch` for appending new rows, creating sheets, formulas, formatting, or layouts without a unique key column or headers in row 1. For direct A1 writes, map each target header and record back to the original grid coordinates, including the header row and any blank rows. Change only the requested cells and preserve all other columns.
