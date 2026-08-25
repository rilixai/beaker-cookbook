# Looking up a row in Google Sheets: locate the spreadsheet, then match exactly

Procedure for tasks that require reading or finding data in a Google Sheet.

1. First list/search available spreadsheets to get the exact spreadsheet and
   worksheet IDs — do not assume names.
2. Read the header row to learn the actual column names before filtering.
3. Use the lookup/search tool with the exact column name and value from the task;
   prefer exact matches, and fall back to scanning rows only for fuzzy criteria.
4. If multiple rows match, re-read the task for a disambiguating field
   (date, status, owner) before picking one.
5. Carry values forward exactly as they appear in the sheet (IDs, emails, amounts) —
   downstream steps are checked against the sheet's literal contents.
