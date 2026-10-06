---
name: slack
description: Procedures for the Slack app (post messages, channels, users, search, etc.).
---

## Reading
- Read with slack_list_channels, then slack_list_channel_messages(channel="#name", limit 100) on the relevant channels.
- slack_find_message returns one message and fails on multi-word, OR, quotes or filters; avoid it. slack_search_messages does not exist.

## Posting
- slack_send_channel_message(channel="#name" or channel ID, text). slack_get_conversation(channel="#name") returns the ID if needed.
- slack_send_direct_message(user=<Slack user ID like U_...>, text). Get IDs from a sheet column or slack_find_user_by_name with the handle "first.last" (a "First Last" full name with a space is not found).
- Text must name each affected entity (person, company, item, ticket) and include required values verbatim. Do not mention skipped/excluded items or unapproved content.
- Post to a channel only when the task's trigger condition holds (e.g. "if status is Done").
