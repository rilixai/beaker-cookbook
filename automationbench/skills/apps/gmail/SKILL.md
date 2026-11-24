---
name: gmail
description: Procedures for the Gmail app (send, find, label, threads, drafts, etc.).
---

For plain-text Gmail messages, pass readable fields through `api_fetch`. The harness encodes the text; no `base64_encode` call is needed.

- Send: endpoint `gmail.users.messages.send`, body `{"to":"recipient","subject":"subject","text":"complete readable body"}`.
- Save an unsent draft: endpoint `gmail.users.drafts.create`, body `{"message":{"to":"recipient","subject":"subject","text":"complete readable body"}}`.
- `to`, `cc` and `bcc` accept strings or lists of strings. `text` contains actual newlines. `body` is an alias for `text`; supply only one.
- For a reply, put the original `threadId` alongside `to`, `subject` and `text` inside the message. A subject beginning with "Re:" does not attach a reply to its thread.
- Read the saved draft with endpoint `gmail.users.drafts.get`, path `{"id":"<returned draft ID>"}`. Check recipient, subject, readable body and reply thread. To correct content, use `gmail.users.drafts.update` with that draft ID in path and the complete message in body. If the reply thread is wrong, recreate the reply in the correct thread and delete the incorrect draft. Keep content already correct and avoid test or duplicate drafts.

For HTML, attachments or mail without a To recipient, native `raw`/`payload` formats remain available. For `raw`, encode the complete email with headers and a blank line before its body; copy the returned encoding exactly. Drafts wrap the native representation in `message`.

Keep drafts as drafts when the task requests review; send only when requested.
