---
name: zendesk
include_in_listing: true
description: Procedures for Zendesk ticket discovery and batch filtering by status, priority, tags, requester, or organization.
---

For Zendesk ticket batches selected by status, priority, tags, requester, or organization, discover and call `zendesk_get_tickets` to read the full ticket list. Reuse a full list already read. Apply the task's retrieved rules to each ticket's fields before deciding which tickets to process.

`zendesk_find_ticket` searches literal text in the subject and description. It does not parse field queries such as `status:open`, and an empty query does not list tickets. Use its `ticket_id` parameter for an exact ticket lookup. A text search returning no tickets does not show that a field-based batch is empty.
