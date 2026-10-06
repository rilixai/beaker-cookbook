---
name: freshdesk
description: Procedures for the Freshdesk app (tickets, contacts, notes, priorities).
---

- freshdesk_get_tickets first; for an existing ticket covering the same customer/issue, add a note (freshdesk_add_note_to_ticket) citing the source ticket ID instead of creating.
- freshdesk_create_ticket(subject, description, status=2 (Open), priority, email=<customer email>, tags). Pass the customer's email (not a source-system contact ID) so the requester contact is created.
- Priority codes: 1 Low, 2 Medium, 3 High, 4 Urgent. Take the number from the policy's mapping table by exact match; use the default row when no exact match.
- Keep subject verbatim from the source ticket; prefix only if the policy says so.
