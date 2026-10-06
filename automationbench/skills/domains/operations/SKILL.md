---
name: operations
description: Procedures and playbooks for operations-domain tasks (syncing status updates from email to boards, picking the next task from a plan sheet, project tools like Asana/Basecamp/Monday/Trello/Pipefy, policy publishing).
---

Operations tasks pair a source (email, sheet) with a board update and a confirmation. When the task mentions an update policy, confirmation format, reference code or required counts and names no sheet, find the config spreadsheet with one Drive fullText search (e.g. "update policy") and read every tab. When the task gives explicit target values (phase, status, field IDs), apply them directly.

## Syncing an update from email to a board
- Pull every message about the item. Apply only messages about the exact item (not its planning phase, branch site, phase 2, or a sibling the task names as out of scope).
- Discard retracted and tentative messages, and other teams' informal corrections unless the update policy names them as authoritative.
- Apply the update policy's rule for which message counts; set the board columns to that message's values exactly.
- Confirmation reply: follow the config format exactly (reference code, required phrases, counts). For "X of N reviewed", N counts every message you reviewed under the config's definition - including retracted, superseded, other-team and sibling-phase messages about the item; leave out only topics the config names as excluded.
- Post completion notices only when the authoritative status really is the trigger status.

## Picking the next task from a plan/queue sheet
- Read the policy tab first (eligible statuses, holds such as disputed/legal, internal vs external).
- Filter rows on all criteria the task gives (phase, task name, type); choose by the stated urgency order (priority, then earliest due/needed-by date).
- Copy the selected row's text verbatim into names/titles and use its date for due dates.

## Urgency alerts
- "Immediate attention" typically = top priority and/or needed-by date already past relative to the task's today. Alert the alert channel only for those.

## Publishing policies (wiki pages + notification)
- Publish only the approved/current version (correct effective date); exclude drafts and superseded versions.
- Create the page the task names (a page-search hit with sample content is mock boilerplate, not an existing page).
- The notification email must include the exact created page title, the effective date and the key change text.

## Decisions from email onto cards/items (approvals, reviews)
- One decision per message; map each to the phase and status values the task lists; confirm each one back to the sender.
- Card/record tools need the system ID (e.g. card_...): if a find-by-title returns nothing, call it with an empty title or list the table to get all records, and match by name.
