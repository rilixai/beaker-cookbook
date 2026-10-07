---
name: jira
description: Procedures for the Jira app (issues, comments, transitions, watchers, attachments, etc.).
---

- jira_create_issue(project=<key>, issuetype=<type exactly as given, e.g. "Task">, summary, description). Pass both project and project_key / issuetype and issue_type when the schema has both.
- Summary should name the entity and reference (e.g. order/ticket number); description carries customer name, email, amount verbatim and the reason from the policy.
- The response key may be a mock value; reference the source record ID in your messages rather than the returned key.
