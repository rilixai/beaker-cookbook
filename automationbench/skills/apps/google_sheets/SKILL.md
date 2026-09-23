---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

This is the operating procedure for the Google Sheets app (rows, lookups, worksheets, updates, etc.).

When a task or policy names a related list or configuration without a worksheet ID, inspect the known spreadsheet's metadata to discover its tabs. Use the returned worksheet IDs and titles to read the relevant policy, exclusion, override, or format rows. If the list is not in that spreadsheet, search Drive for its title and read the matching spreadsheet. Do not guess several tab names or assume the transaction tab contains every rule.

Read the cells of each required tab; a spreadsheet title or tab listing is not its contents. Preserve the column names when interpreting rules so the match uses the right field. Read every column used by a condition or exception, including notes. Reuse rows already read, and fetch further rows only when the result indicates more remain.
