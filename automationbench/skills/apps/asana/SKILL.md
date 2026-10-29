---
name: asana
description: Procedures for Asana tasks, native task IDs, project sections, and tags.
---

When a task requires a project section or tag, resolve the task and destination IDs before assigning them.

For a named section, use the workspace and project IDs supplied by the task. If either is unknown, call `asana_find_section` with the section name and an empty string for each unknown filter. Empty workspace and project filters are omitted by this lookup. Read the matching result's `workspace`, `project`, and `section` values. An empty project list does not show that the section is absent. Use the resolved workspace and project when creating the task.

Save the created task's `gid` from its matching `results` entry. For section and tag assignments, use this `gid` as `task_id`. The entry's `id` identifies the action record; `invocation_id` and `response_uuid` identify the request. Keep the workspace and project IDs resolved from the task or section lookup when the creation response includes other workspace or project metadata.

Apply a requested section with `asana_add_task_to_section`, using the saved task `gid` and resolved workspace, project, and section IDs. Apply a requested tag with `asana_add_tag_to_task` on that same task `gid`. Check the returned assignment arguments before reporting completion.
