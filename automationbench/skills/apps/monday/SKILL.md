---
name: monday
description: Procedures for the Monday.com app (boards, items, status and date columns).
---

- monday_board_items may return nothing; use monday_find_item(board_id, name=<exact item name>) to get item_id.
- monday_change_status_column_value(board_id, item_id, column_id, value_label=<label text, e.g. a status label exactly as the source writes it>).
- monday_change_date_column_value(board_id, item_id, column_id, value_date YYYY-MM-DD).
- column_id must be exactly the column name the task gives ("status", "due"); do not invent variants like "due_date". Calls succeed with any column_id, so a wrong one silently fails grading.
- Change only the item the task targets; similarly named items (planning, phase 2, other sites) are different items.
