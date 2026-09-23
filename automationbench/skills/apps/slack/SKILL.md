---
name: slack
description: Procedures for the Slack app (post messages, channels, users, search, etc.).
---

This is the operating procedure for the Slack app (post messages, channels, users, search, etc.).

For a private notification, use the direct-message action and the recipient's Slack user ID. If a directory or tracker supplies both an email and a Slack user ID, use the Slack ID for Slack and the email for Gmail. A user ID is not a channel ID. Resolve an email to a Slack user only when no Slack ID is already available. Check the send result; a failed email lookup does not invalidate an ID supplied by the workflow's directory. Retry a failed send with the confirmed ID after checking that no message was delivered.
