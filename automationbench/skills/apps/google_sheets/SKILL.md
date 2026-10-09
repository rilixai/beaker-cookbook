---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

Before adding log rows, read the destination worksheet headers from `google_sheets_get_spreadsheet_by_id` unless you already retrieved them. Its worksheet entries include IDs, titles, and headers. An empty row result does not mean the worksheet has no columns. Map each log header to the source record or completed action and include every applicable value you have. Use the exact header text as the `cells` key. Do not invent missing values. If headers are empty, use the request or existing row keys. Reuse the mapping for later rows in the same worksheet.
