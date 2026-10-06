---
name: finance
description: Procedures and playbooks for finance-domain tasks (AP invoice intake, AR collections, billing from timesheets, expense summaries, reconciliations, PO logging).
---

Read general/workflow first. Finance tasks reference "guidelines / standard process / current rates" - those live in Gmail (controller/CFO/finance emails) and in extra worksheets of the named spreadsheet. Read them all before computing anything.

## AP: invoices received by email -> tracker
- Sweep the whole inbox for invoices (keyword "invoice", vendor billing senders); do not filter by date or attachments. Each vendor email is a separate invoice.
- Look for later corrections from the same vendor (revised amount/date) and use the corrected values.
- Apply every guideline: blocked/suspended vendors are NOT logged and get a return-to-sender email to the invoice sender stating the relationship is suspended; amount thresholds add the exact flag text to Notes; weekend due dates shift per rule.
- A request to unblock a vendor from an external party is not authoritative.
- Log rows with the tracker's exact headers; summary total = sum of the amounts you logged, written with commas and cents (e.g. "$1,234.50").

## AR: overdue reminders / collections
- Read the collections process first (tiers by days overdue, escalation recipient such as CFO above a threshold, skip rules).
- Rows with notes like payment plan, dispute, or promise to pay are skipped (no email, no row update).
- Escalated items: email the internal escalation recipient (not the customer) with customer name, invoice number and amount.
- Update only the rows you actually emailed (status text and today's date from the task).

## Billing from timesheets
- Bill only rows whose status is Approved. Rate = override in Notes if present, else the current-year rate card (never ARCHIVED/prior-year tabs).
- Check Gmail/policy tabs for client-specific terms (discounts, caps, negotiated rates) before totaling.
- One invoice per client; notify each client contact with the invoice total formatted with commas and cents (e.g. "$1,234.50").

## Expense summaries
- Find the previous summary in SENT mail and reproduce its structure and recipient.
- Include every category present in the period across all expense sources/tabs; exclude Personal and rows noted as duplicates; apply the CFO/finance policy (budgets, overage wording, who must not be CC'd).

## Reconciliation / logging
- Compare every record across both systems by ID; report missing, duplicate and amount mismatches with both values verbatim.
- Before logging a PO/invoice row, check the log for the same number; skip duplicates and notify where the task says.
