---
name: asana
description: Procedures for the Asana app, including task identifiers for section and tag actions.
---

When an Asana task result contains both `gid` and `id`, use the task's `gid` as `task_id` for `asana_add_task_to_section` and `asana_add_tag_to_task`. Keep that task `gid` for later section and tag actions; do not replace it with an `id` returned by a link action. If the task result has no `gid`, use the task ID supplied by the user or returned by the task lookup. This identifier rule applies to section and tag actions; keep the identifiers used for task updates unchanged.
