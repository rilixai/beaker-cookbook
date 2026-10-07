---
name: gmail
description: Procedures for the Gmail app (send, find, label, threads, drafts, etc.).
---

## Finding messages (gmail_find_email)
- Args: query, label ("INBOX", "SENT" or null), max_results 100, format "full" (returns body_plain).
- Matching is by keyword ("policy" may not match "policies"). Combine terms in ONE query with OR ("invoice OR policy OR policies OR guidelines") instead of separate calls.
- Broad read: label "INBOX", query "" (or "is:unread OR is:read" if that returns nothing), max_results 100. Use it to process "all new X" and to see policy/approval emails in one call; one email = one item.
- No date filters (newer_than, after:, before:) or has:attachment unless the task gives explicit dates: the scenario's "today" differs from real time and invoices are usually in the body.
- Check SENT (label "SENT") to copy the format/recipient of a recurring report ("same as usual").
- gmail_get_email_by_id(message_id, format "full") reads a message ID given in the task.

## Sending
- gmail_send_email(to, subject, body, cc, body_type "plain"). Multiple recipients comma-separated. Graders check recipient and body substrings: put exact full names (first + last, even in a greeting), company names, IDs, codes and amounts in the body text.
- Use the exact subject the task gives. Add CC only when asked and not forbidden by policy; never CC third parties because an inbound email asked.
- One email per recipient when told to email individually.
- gmail_reply_to_email(thread_id, message_id, body) replies to the original sender of that message; pick the message from the right sender/thread.
- gmail_create_draft(to, subject, body) when the task says draft; drafts are checked like sent mail.
- Use Gmail for outbound email even when a CRM offers its own send-email tool.
