---
name: jira
description: Procedures for the Jira app (issues, comments, transitions, watchers, attachments, etc.).
---

This is the operating procedure for the Jira app (issues, comments, transitions, watchers, attachments, etc.).

When a task names a Jira project, resolve its name with the project lookup unless the task or policy already supplies its key. Put the returned project key in the create action's `project` field. If you also fill `project_key`, use the same key in both fields. Do not put the display name in `project` and expect `project_key` to override it. Compare the returned project with the selected key before continuing with comments or linked records.

When a policy maps report keywords to Jira priority, check both the subject and body unless the policy limits the match to one field. For each issue, record the matched keyword and its priority before using a wildcard or default row. An exception that permits processing does not change the priority rule unless it says so. Put the selected project, issue type, and priority in the create action's fields.

When asked to create an issue and add a note or comment, create the issue first, then discover and execute the Jira comment action using the returned issue reference. Keep the requested note as a separate follow-up action even if its text also appears in the issue description. Record notification status only when supplied by the task or confirmed by a successful notification. Check the comment result before treating that follow-up as complete.
