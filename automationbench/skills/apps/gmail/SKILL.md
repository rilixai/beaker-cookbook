---
name: gmail
description: Procedures for the Gmail app (send, find, label, threads, drafts, etc.).
---

When processing a batch of incoming emails, keep the sender, label, read-state, and exact subject filters supplied by the task or retrieved workflow. Do not add guessed topic words to define the batch: relevant messages may use different wording, even when the search returns matches.

If no mailbox filters are supplied, use `gmail_find_email` with `label="INBOX"` and an empty `query`. Request `format="full"` and `max_results=100`. If `has_more` is true, increase `max_results` until the batch is covered. Reuse a complete batch already read.

Read each message's subject and body to decide relevance. Apply the task's dates, eligibility, duplicate checks, and special handling rules before writing or sending; retrieving a message does not make it eligible.
