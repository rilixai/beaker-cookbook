# v3: breadth pass over the long tail (Step 4.5)
Round-3 categorization of v2 train losses (8.06 total over 71 rows): not-in-observed-data / possible grader issue 3.20 (finance: invoice_email_extract, timesheet_to_invoice, weekly_expense_summary — not fixable by skills); variance 1.67; dropped per-item action 1.22; structural (Slack OR searches, apps/slack unread) 0.84; domain misroute 0.68; same-issue-different-contact gap 0.45. No single behavior clears the ±0.10 test noise floor, so this round bundles four independent minimal fixes:
1. workflow §4: final per-item checklist — every qualifying item gets each requested action exactly once; support: qualifying ticket with failed lookup still gets the requested reply.
2. workflow §1: Slack = list channels and read; find_message takes one plain keyword.
3. support/freshdesk: existing ticket on the same issue from a different contact -> note, not duplicate.
4. system.md: read both domain skills when a task fits two domains.
