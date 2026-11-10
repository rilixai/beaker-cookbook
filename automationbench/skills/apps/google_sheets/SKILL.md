---
name: google_sheets
description: Procedures for the Google Sheets app (rows, lookups, worksheets, updates, etc.).
---

When a spreadsheet provides workflow data, get its worksheet list once with `google_sheets_get_spreadsheet_by_id` and `includeGridData=false` before deciding what to change. The result includes worksheet IDs, titles, and headers. Reuse a worksheet list already returned.

Read the named data tab and any tabs whose titles match the task's policies, updates, exclusions, or needed reference data. Follow references from those tabs. A named data tab can be only one part of the workflow. Select tabs by their relevance to the task.

Use the actual worksheet ID from the list for row reads. `google_sheets_find_worksheet` looks up a title. If a guessed title fails or a guessed tab is empty, inspect the worksheet list before trying more names or concluding that facts are missing.

When a policy table maps keywords to categories, test the configured keywords against the source text the policy covers. Match without regard to case unless the policy says otherwise. Collect all matching rules before choosing a category, then resolve overlaps using the policy's stated precedence. Do not invent synonyms or use the topic's meaning as a keyword match unless the policy allows it. Use the resulting category for its mapped actions, routing, priority, and comments.

When a workflow mapping requires a source event or completion marker, plan it as a separate app write. For each eligible record, complete the mapped destination write first, then use the source record's ID to record the configured marker through its app tool. Discover that tool if needed. Check that both writes succeeded before reporting the record complete; putting the marker in a destination description or recap does not record it in the source app. Do not record completion for excluded records or failed destination writes.
