---
name: support
description: Procedures and playbooks for support-domain tasks (helpdesk migrations and escalations, refunds, order lookups, callback scheduling, org/company syncs).
---

Read general/workflow first. Support tasks are driven by a policy spreadsheet with several tabs (criteria, priority mapping, exclusion tags, overrides). Read every tab, then decide per ticket.

## Deciding per ticket
- Qualify on the primary criteria (status, tag) AND the other fields: classification/type, the customer's actual message text (a return or question is not a refund request), order status (already refunded/cancelled), notes in side sheets.
- Override rules: "Never/hold" rules (legal review, acquisition pending) block the item; "Always" rules force inclusion even when an exclusion tag is present. Match override values exactly as the rule describes (tag, email domain contains, company name).
- Priority mapping: exact tag/priority match; use the default row when nothing matches exactly.
- Items that fail are not logged, not replied to and not mentioned, unless the task defines an outcome value for them (e.g. Not Found, Expired, Denied) - then log/reply with that outcome only.

## Creating tickets in another system (migration / escalation)
- First list existing tickets in the target. If one already covers the same customer/issue, add a note to it containing the source ticket ID instead of creating a duplicate.
- Create with subject verbatim, description from the source, mapped priority, and the customer's EMAIL (look it up in the source contacts) so the requester/contact is created.
- Comment/message on each source ticket with the exact phrase the task gives; log each migrated item in the log sheet.

## Refund processing
- Per ticket: find the order; check status, refund window from order date to the task's today, category eligibility, thresholds (standard vs VIP list), override rules (flags, loyalty emails), and that the requester email matches the order's customer.
- Outcomes: draft confirmation (gmail_create_draft) vs escalate (Jira issue) as the policy dictates; log one row per processed ticket; reply on the ticket with the outcome word the task specifies.

## Callback scheduling
- Eligible = requested status AND callback-type classification, minus contacts excluded in the notes sheet.
- Check the calendar for existing events of the same contact; never double-book a contact or overlap slots.
- Event title "<Customer full name> - <Ticket subject>", attendee = contact email; then add a ticket comment like "Callback scheduled for <date time>".

## Company / org sync
- List target companies first; reuse a name match instead of creating. Every source org is synced unless a rule excludes it; unmapped tags just get no tag.
- Apply tag mapping exactly; apply Always/Never overrides; the recap lists the names of synced companies.
