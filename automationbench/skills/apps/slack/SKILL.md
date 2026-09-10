---
name: slack
description: Procedures for the Slack app (post messages, channels, users, search, etc.).
---

# Slack

## Always read Slack, even when the task does not mention it
Slack channels are where colleagues post last-minute corrections and overrides
(corrected amounts, negotiated discounts, reclassifications, "skip X", changed
recipients). These override the figures found in emails and spreadsheets.

Procedure at the start of a task (in parallel with other reads):
1. `slack_list_channels` - list every channel.
2. `slack_get_channel_messages` (alias `slack_list_channel_messages`) with
   `limit: 50` on every channel whose name relates to the task's domain (e.g.
   `accounts-payable`, `billing`, `finance-internal`, `procurement`, `sales`, `hr`,
   `support`, `ops`). Also read `general` if the workspace is small.
3. For parents with `reply_count > 0`, call `slack_get_thread_replies`.
4. Note any message that names an entity from the task (vendor, client, employee,
   invoice, PO) and apply it when computing values.

`slack_find_message` only returns the single best match - prefer listing channel
messages so nothing is missed.

## Posting
- `slack_send_channel_message` accepts the channel name (`procurement`) or ID.
- Include the identifiers and entity names the task asks for verbatim (PO numbers,
  invoice numbers, vendor/customer names, amounts with original formatting).
- Post one clear message per required notice; do not post extra chatter or summaries
  of skipped items unless the task asks for them.
