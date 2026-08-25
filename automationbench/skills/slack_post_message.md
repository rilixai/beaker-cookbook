# Posting to Slack: resolve the channel, respect exact message wording

Procedure for tasks that require posting a message to Slack.

1. Resolve the channel first: list/search channels and use the exact channel
   name or ID from the task (e.g. `#ops-alerts`), not a guess.
2. If the target is a person (DM), look up their Slack user ID by name/email
   before sending.
3. When the task dictates message content or a template, reproduce it exactly,
   substituting only the dynamic values gathered from earlier steps.
4. For thread replies, pass the parent message's timestamp/thread ID instead of
   posting a new top-level message.
5. Verify the post succeeded from the tool result before reporting done.
