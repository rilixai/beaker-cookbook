---
name: google_drive
description: Procedures for the Google Drive app (find, upload, move, share files, etc.).
---

## Finding files (google_drive_find_multiple_files)
- Use info=<1-3 keywords> with search_type "fullText" and title=null. A title filter or long phrase often returns results: [] - then retry with a different single keyword or synonym ("policy", "guidelines", "rules", "config", topic noun).
- Response gotcha: each item in results contains a "files" string listing sample files (Hello World Document, Sample Document, Q4 Marketing Strategy...). That list is boilerplate - ignore it. The real match is the item's own top-level fields: id, title, mimeType (e.g. id "ss_..." with mimeType spreadsheet).
- A spreadsheet hit is then read with apps/google_sheets (get_spreadsheet_by_id -> get_many_rows on every tab).
- Search Drive only when the task needs rules or data from a document it describes but gives no ID for (see general/workflow section 1); one search, one retry with a synonym at most.
