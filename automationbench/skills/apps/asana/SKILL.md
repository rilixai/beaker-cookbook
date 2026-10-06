---
name: asana
description: Procedures for the Asana app (tasks, sections, tags, projects).
---

- A task reference like "ws_x/proj_y" means workspace "ws_x" and project "proj_y": pass project "proj_y", never the combined string (lookups with the combined string return nothing).
- Resolve the section ID: asana_find_section(workspace, project="proj_y", name=<section name>) -> use the returned "section" value (e.g. "sec_...") as the section argument.
- asana_create_task(workspace, project, name, due_on / dueDate YYYY-MM-DD, tags). The task ID for all follow-up calls is results[0].gid (a long number) - NOT the "id" field ("asana_...").
- Always also call asana_add_task_to_section(task_id=<gid>, workspace, projects="proj_y", section=<section id>) and asana_add_tag_to_task(task_id=<gid>, tag=<tag name>), even if tags were passed at creation.
- Follow-up responses show boilerplate ("Example Task", 2024 dates); ignore them.
