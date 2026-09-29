# v2: per-record screening (exclusion by Status/Notes/cross-tab, blanks, units, dates)

Hypothesis: after v1 the agent finds the rules but still selects the wrong records. It over-includes rows disqualified by a Status/Notes value or a cross-tab lookup, and it mis-normalizes units/denominators/dates. A short explicit screening procedure will fix this.

Evidence (v1/train):
- hr.benefits_enrollment_audit 0.83: row Status "Inactive" (Felix Braun) included in Slack + email.
- support.hiver_workload_forecast 0.65: Status "pto" agent logged in forecast; utilization written as "100.00%"; subject lacked "overload".
- support.gorgias_fraud_detection 0.72: flagged VIP Diamond/Gold (ORD-7703, 7705) and Under Review order (7711).
- operations.utility_cost_allocation 0.50: ignored Notes "temporarily vacant" row; denominator wrong (5666.67 vs 4533.33); mentioned "Vacant" in email.
- operations.calendar_slack_training 0.17: picked Tentative session, not the Confirmed/available one.
- marketing.content_scoring 0.91: Draft page included.
- sales.unreliable_label_account_review 0.86: blank-data account listed.
- finance.expense_policy_violation 0.56: $320 / 2 nights flagged vs $250/night (per-unit not divided).
- hr.contractor_renewal 0.38: 60-day window / expired branch wrong (date anchor).
Approx pc lost in these ~9 cases: 3.0/36 = 0.083 pc; realistic capture ~40% => ~+0.03 score.

Change: prompts/system.md only, +8 lines (section "Screen every record before acting"). No skill bloat.
Risks: over-exclusion (Status "Under review"/"cleared" wording may be legit trigger in some tasks, e.g. support tickets pending); "latest dated item" as today may mis-anchor.
