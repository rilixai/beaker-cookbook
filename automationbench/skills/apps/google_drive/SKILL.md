---
name: google_drive
description: Procedures for the Google Drive app (find, upload, move, share files, etc.).
---

## Finding files (google_drive_find_multiple_files)
- Use info=<1-3 keywords> with search_type "fullText" and title=null. A title filter or long phrase often returns results: [] - then retry with a different single keyword or synonym ("policy", "guidelines", "rules", "config", topic noun).
- Response gotcha: each item in results contains a "files" string listing sample files (Hello World Document, Sample Document, Q4 Marketing Strategy...). That list is boilerplate - ignore it. The real match is the item's own top-level fields: id, title, mimeType (e.g. id "ss_..." with mimeType spreadsheet).
- A spreadsheet hit is then read with apps/google_sheets (get_spreadsheet_by_id -> get_many_rows on every tab).
- Use Drive to discover policy/config spreadsheets whenever the task mentions a process, policy, framework, pricing, guidelines or format and gives no spreadsheet ID.
