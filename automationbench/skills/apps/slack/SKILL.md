---
name: slack
description: Procedures for the Slack app (post messages, channels, users, search, etc.).
---

`slack_list_channel_messages` and its alias `slack_get_channel_messages` accept a channel ID or name. `slack_list_channels` returns channel metadata, not message contents. Reuse channel history already read in this task unless the relevant state changed.

`slack_find_message` uses a case-insensitive literal substring match, not a keyword or Slack query parser. Use a short unquoted phrase or one record identifier. Do not combine unrelated terms, quote the phrase, or add search operators. An empty result only means that exact text was absent. Use the channel-history check required by the workflow before concluding that no update exists.
