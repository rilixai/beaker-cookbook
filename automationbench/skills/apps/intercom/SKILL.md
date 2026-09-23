---
name: intercom
description: Procedures for the Intercom app (contacts, conversations, tickets, tags, notes, etc.).
---

This is the operating procedure for the Intercom app (contacts, conversations, tickets, tags, notes, etc.).

When a policy selects Intercom conversations by tag or state, discover intercom_get_conversations and list conversations before applying those fields. A text search for the workflow topic can omit tagged requests whose title uses different words. Match the exact conversation tag and state, then join its contact IDs to contacts and their associated companies. Keep conversation tags, contact tags, and company fields separate.

For scheduling, give each conversation a decision before creating events: skip under an explicit scope or duplicate rule, book if qualified, or send the prescribed decline. Check the contact type and email domain, the linked company's size, and any explicit override against the retrieved policy. Record the actual values and the rule that supports the decision. An exception changes only the conditions it overrides. Derive the event fields from the scheduling rules after qualification. Before booking in Google Calendar, read `apps/google_calendar` to carry those fields into the event.

Keep decisions keyed by conversation and contact ID across all batches. A later event call must still pass the same qualification checks; a previous decline is not permission to book. Track successful bookings by contact ID so another thread cannot create a second booking. Follow explicit skip rules for closed conversations, already scheduled contacts, and duplicate requests. Do not send a decline to a skipped conversation unless the policy requires one. Send each required outcome once and tag a contact as scheduled only after its event succeeds.
