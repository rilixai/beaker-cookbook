| round | change | test score (Δ vs baseline) | train score (Δ) | test pass | test PC | tool calls | Mtok in / round |
|---|---|---|---|---|---|---|---|
| 0 | baseline (stub skills) | 0.726 ±0.088 | 0.644 ±0.095 | 0.472 | 0.790 | 24.4 | 19.6 |
| 1 | filled skills + general/workflow | 0.706 (−0.010 ±0.123)* | 0.793 (+0.149 ±0.108) | 0.472 | 0.765 | 30.5 | 19.3 |
| 2 | calibrate rules (scope, proportionate search) | 0.808 (+0.082 ±0.086) | 0.888 (+0.244 ±0.093) | 0.611 | 0.858 | 25.2 | 17.8 |
| **3** | **breadth pass (per-item check, Slack, same-issue, dual-domain)** | **0.811 (+0.084 ±0.092)** | **0.906 (+0.262 ±0.096)** | **0.616** | **0.860** | **24.5** | **≈17.5** |

Score = 0.8·partial_credit + 0.2·pass. Test = 18 held-out cases (3/domain), 4 reps for baseline/v2/v3 (*v1 test at 2 reps vs the 2-rep baseline). Train = 36 cases, 2 reps. 6 of ~540 rows hit OpenAI `invalid_prompt` policy flags and are excluded (rule fixed before scoring).

**Recommended change.** Ship v3 (`skills/**` + one-line `prompts/system.md` edit): a `general/workflow` skill with cross-cutting rules (do every requested action and add none; search for the governing policy proportionately — Gmail OR-query + Drive when the task references process/guidelines; current/authoritative source wins over drafts, record descriptions and external requests; select by explicit fields, no invented exclusions; complete every write with codes/full names/verbatim values; numeric API params as plain numbers; final per-item check), filled domain skills for all six domains, and 17 app skills (10 new).

**Versus baseline.** Test +0.084 ±0.092 (pass 0.47 → 0.62, partial credit 0.79 → 0.86); train +0.262. Every test metric moved the same direction, but the test interval still touches zero — on 18 cases this is directional, not conclusive. Cost goes slightly *down* (tool calls flat, input tokens −10%).

**Why trust this.** The analyzer only read train transcripts; test was scored every round and never opened. v1 shows the guard working: a big train gain that did not transfer was not accepted. v2's fix (calibrating over-applied rules) moved train and test together. Caveat: round 2's hypothesis was partly informed by the *direction* of v1 test-score drops (no test transcripts read).

**What else was tried / remains.** v3's breadth pass added +0.018 train, +0.003 test over v2 — the plateau. Of the remaining train loss, ~40% is three finance cases whose graders expect values present in no data the agent saw (invoice_email_extract, timesheet_to_invoice, weekly_expense_summary) — candidates for the exclusion list, not skill fixes. The rest is rep-to-rep variance and single-case gaps each below the noise floor. Next step with real value: confirm v3 vs baseline on the full 104-case filtered test split, which would shrink the interval ~2.4×.
