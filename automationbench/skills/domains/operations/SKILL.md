---
name: operations
description: Procedures and playbooks for operations-domain tasks (syncing status updates from email to boards, picking the next task from a plan sheet, project tools like Asana/Basecamp/Monday/Trello/Pipefy, policy publishing).
---

Read general/workflow first. Operations tasks pair a source (email, sheet) with a board update and a confirmation; there is usually an update policy and a confirmation format in a config spreadsheet (find it via Drive full-text search, e.g. "update policy", "confirmation format").

## Syncing an update from email to a board
- Pull every message about the item. Keep only messages about the exact item (not its planning phase, branch site, phase 2, or a sibling item named in the task as out of scope).
- Discard retracted and tentative messages, and other teams' informal corrections unless the update policy names them as authoritative.
- Apply the update policy's rule for which message counts; set the board columns to that message's values exactly.
- Confirmation reply: follow the config format exactly (reference code, required phrases, counts such as "X of N reviewed" where N = every candidate message you reviewed for that item).
- Post completion notices only when the authoritative status really is the trigger status.

## Picking the next task from a plan/queue sheet
- Read the policy tab first (eligible statuses, holds such as disputed/legal, internal vs external).
- Filter rows on all criteria the task gives (phase, task name, type); choose by the stated urgency order (priority, then earliest due/needed-by date).
- Copy the selected row's text verbatim into names/titles and use its date for due dates.

## Urgency alerts
- "Immediate attention" typically = top priority and/or needed-by date already past relative to the task's today. Alert the alert channel only for those.

## Publishing policies (wiki pages + notification)
- Publish only the approved/current version (correct effective date); exclude drafts and superseded versions.
- The notification email must include the exact created page title, the effective date and the key change text.

## Vendor/approval decisions from email
- One decision per message; map each decision to the phase and status field values the task lists; confirm each one back to the sender.
