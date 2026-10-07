---
name: zoho_desk
description: Procedures for the Zoho Desk app (tickets, contacts, comments, status updates).
---

- zoho_desk_get_tickets returns status, priority, classification, contact_id, subject. Filter on status AND classification/other fields the policy names.
- zoho_desk_get_contacts maps contact_id -> first_name, last_name, email (needed for customer names, calendar attendees and cross-system tickets).
- zoho_desk_add_comment(ticket_id, content) is how a ticket is "updated"/"commented": write the action and key detail (e.g. "<Action> scheduled for <date time>", "<required phrase> - <target ticket id>"), using the exact phrase the task gives.
- zoho_desk_update_ticket changes fields (status, priority); use it only when the task asks for a field change, in addition to the comment.
- Comment only on tickets you actually acted on.
