---
name: jira
description: Procedures for the Jira app (issues, comments, transitions, watchers, attachments, etc.).
---

For `jira_create_issue`, put the project key in `project`. When the task gives a project display name, use `jira_project` to resolve it and use the returned `project` value. Reuse a key already supplied by the task or a lookup. `project_key` is only a fallback when `project` is empty; it does not override a display name in `project`. Leave `project_key` null when using `project`.
