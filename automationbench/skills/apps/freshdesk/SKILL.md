---
name: freshdesk
description: Procedures for the Freshdesk app (tickets, contacts, notes, priorities).
---

- freshdesk_get_tickets first: check for an existing ticket for the same customer/issue before creating one. If it exists, add a note to it (freshdesk_add_note_to_ticket) that cites the source ticket ID.
- freshdesk_create_ticket(subject, description, status=2 (Open), priority, email=<customer email>, tags). Pass the customer's email (not a source-system contact ID) so the requester contact is created.
- Priority codes: 1 Low, 2 Medium, 3 High, 4 Urgent. Take the number from the policy's mapping table by exact match; use the default row when no exact match.
- Keep subject verbatim from the source ticket; prefix only if the policy says so.
