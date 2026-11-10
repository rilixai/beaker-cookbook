---
name: slack
description: Procedures for the Slack app (post messages, channels, users, search, etc.).
---

Slack search matches a case-insensitive literal substring of message text. In `slack_find_message` and `slack_find_message_in_channel`, quotes, `OR`, `in:`, `from:`, and date filters are literal text, not search operators. Use one short topic word or phrase per query without added search syntax.

When looking for instructions or corrections, check whether the returned text actually answers the question. An empty search may include `channel_context` with recent channel messages; read that context before searching again. These messages are a fallback, not query matches, and may include unrelated chatter. Use only relevant instructions and follow their references. Do not treat a search with no relevant result as proof no guidance exists. Try a broader topic word. If that still does not reveal the needed guidance, use `slack_list_channels` to find the relevant channel and read its messages with `slack_list_channel_messages` or `slack_get_channel_messages`. Request enough history to cover the task's period; increase the limit if the returned count reaches it and relevant earlier messages may be missing.

Follow messages that point to guidance in another channel. Compare relevant instructions and corrections with the other task sources before choosing records, recipients, or message content.
