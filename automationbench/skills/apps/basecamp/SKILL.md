---
name: basecamp
description: Procedures for the Basecamp 3 app (todos, todo lists, projects).
---

- basecamp3_todo(account, project, todo_set, todo_list, content, due_on YYYY-MM-DD).
- A reference like "acct_x/proj_y, set_z/list_w" maps to account "acct_x", project "proj_y", todo_set "set_z", todo_list "list_w" - split the paths, never pass "acct_x/proj_y" as project.
- content is checked verbatim: build it exactly in the format the task gives from the selected row's values; set due_on from the same row.
