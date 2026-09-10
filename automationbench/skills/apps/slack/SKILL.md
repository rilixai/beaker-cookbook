---
name: slack
description: Procedures for the Slack app (post messages, channels, users, search, etc.).
---

# Slack

## Always read Slack, even when the task does not mention it
Colleagues post last-minute corrections and overrides in Slack (corrected amounts,
discounts, reclassifications, skip instructions, changed recipients). These override
figures found in emails and spreadsheets.

At the start of a task, in parallel with other reads:
1. `slack_list_channels`.
2. `slack_get_channel_messages(channel, limit=50)` on every channel that could relate to
   the task's domain.
3. `slack_get_thread_replies` for any parent with replies.
4. Apply any message that names an entity from the task when computing values.

`slack_find_message` returns only the single best match; prefer listing channels.

## Posting
- `slack_send_channel_message` accepts the channel name or ID.
- Include the identifiers, entity names and amounts the task asks for verbatim.
- One clear message per required notice; no extra chatter.
