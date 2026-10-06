---
name: support
description: Procedures and playbooks for support-domain tasks (helpdesk migrations and escalations, refunds, order lookups, callback scheduling, org/company syncs).
---

Support tasks are driven by the policy/lookup spreadsheet the task names (criteria, priority mapping, exclusion tags, overrides, notes); if it is only described, find it with one Drive fullText search. Read every tab; that is the rule source, so no Gmail policy search is needed unless the task points there.

## Deciding per ticket
- Qualify on the task's primary criteria (status, tag) and the explicit fields the sheet defines (classification/type, order status, Eligible, Flags, side-sheet exclusions). A ticket whose message is a different request type (a return or question is not a refund request) does not qualify.
- Override rules: "Never/hold" rules block the item; "Always" rules force inclusion even when an exclusion tag is present. Match override values exactly as the rule describes (tag, email domain contains, company name, Flags field).
- Priority mapping: exact tag/priority match; use the default row when nothing matches exactly.
- Items that fail are not logged, not replied to and not mentioned, unless the task defines an outcome value for them (e.g. Not Found, Expired, Denied) - then log/reply with that outcome only.

## Creating tickets in another system (migration / escalation)
- First list existing tickets in the target. If one already covers the same customer/issue, add a note to it containing the source ticket ID instead of creating a duplicate.
- Create with subject verbatim, description from the source, mapped priority, and the customer's EMAIL (look it up in the source contacts) so the requester/contact is created.
- Comment/message on each source ticket with the exact phrase the task gives; log each migrated item in the log sheet.

## Refunds / order lookups
- Find the order by the number in the ticket. Not Found = the number is absent from the order data.
- Decide with the sheet's explicit rules only: order status, refund window (order date to the task's today), category Eligible column, amount threshold (VIP threshold when the requester's email is on the VIP list), override and repeat/flag lists. A threshold is inclusive ("up to $X") unless the sheet says otherwise; above it = escalate.
- Descriptive category notes ("must be unworn") and a requester email that differs from the order's customer are not grounds to deny or mark Not Found.
- Outcomes: draft confirmation (gmail_create_draft) vs escalate (Jira issue) as the policy dictates; log one row per processed ticket; reply on the ticket with the outcome word the task specifies.
- Order-status replies answer the ticket's question with the sheet's values (status, tracking, ship date).

## Callback scheduling
- Eligible = requested status AND callback-type classification, minus contacts excluded in the notes sheet.
- Check the calendar for existing events of the same contact; never double-book a contact or overlap slots.
- Event title "<Customer full name> - <Ticket subject>", attendee = contact email; then add a ticket comment like "Callback scheduled for <date time>".

## Company / org sync
- List target companies first; reuse a name match instead of creating. Every source org is synced unless a rule excludes it; unmapped tags just get no tag.
- Apply tag mapping exactly; apply Always/Never overrides; the recap lists the names of synced companies.
