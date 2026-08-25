# Sending an email with Gmail: find the right tool, required fields, and verify

Procedure for tasks that require sending an email through the Gmail integration.

1. Search for the Gmail send tool (e.g. `gmail_send_email`) before guessing tool names.
2. Resolve the recipient's actual email address first — if the task names a person,
   look them up (contacts, CRM, or a provided sheet) rather than inventing an address.
3. Required fields typically include `to`, `subject`, and `body`. Copy any exact
   wording the task specifies verbatim into the subject/body.
4. If the task asks to CC/BCC or reply in a thread, set those fields explicitly;
   do not send a fresh email when a reply is requested.
5. After sending, confirm the tool result indicates success before reporting done.
