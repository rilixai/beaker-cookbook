---
name: gmail
description: Gmail procedures for exact recipients and content, send-versus-draft choice, threads, labels, and verification.
---

# Gmail procedure

Treat recipient, message state, subject, body, thread, and attachments as separate requirements. Business records and the user's request decide the intended content; Gmail only transports it.

## Before writing

1. Build the complete eligible recipient set from current business evidence. Check result counts or paging signals, deduplicate by the authoritative email address, and apply exclusions before composing anything.
2. Make a per-recipient composition table containing the exact destination, intended state, subject, required body facts, and prohibited facts. Bind every name, company, amount, date, identifier, offer, or source phrase to the same entity that supplied the recipient address. Preserve requested source text verbatim; do not shorten a full name to a first name when the full name is required.
3. Search for the exact Gmail action and inspect its schema. Distinguish `send`, `draft`, and `reply`: use a send action for requests to email, notify, contact, distribute, or perform outreach. Save a Gmail draft only when the request explicitly asks for an unsent draft. Wording such as "draft a brief" describes authoring the content and does not by itself request Gmail draft state. Reply only when the request requires continuing an observed thread.
4. If the task forbids duplicates or prior outreach, search sent mail and drafts using recipient plus a stable subject or campaign marker before writing. Never send to excluded, suppressed, withdrawn, inactive, or already-contacted recipients when current evidence makes them ineligible.

## Compose and execute

- Put every required fact in the actual subject/body of the artifact sent to that recipient. A final assistant summary does not satisfy message content.
- Keep exact recipient-specific values together; never reuse another entity's name, offer, amount, or status. Preserve punctuation and precision where the request requires verbatim source values.
- For independent recipients, prepare one message per required recipient and execute only after the set and content table are complete. For dependent work such as replying to a found thread or attaching a newly created file, wait for the earlier result and use its observed ID in a later turn.
- Omit optional `cc` and `bcc` recipients unless business evidence requires them. Do not turn a requested draft into a sent message or a requested send into a draft.

## Verify

Inspect every mutation result. A structured error, missing message ID, or unexpected label/state is not success. Confirm that the number of successful messages equals the intended recipient count and that each result reports the expected address, subject, and sent/draft state. Re-read the exact message when a result does not echo enough content to verify required body facts. Repair only the missing or failed artifact; do not resend successful messages.
