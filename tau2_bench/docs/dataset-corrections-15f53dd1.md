# Four supported corrections for run 15f53dd1

Only these four fixes are retained. The 18 customer-behavior clarifications from the earlier draft have been removed completely. Every other effective English task definition remains identical to the pinned upstream data.

| Sample | Retained fix |
|---|---|
| retail-20 | Fix the shared item-modification tool to use each replacement's own catalog price and options. Reference outcomes are rebuilt by replay with the corrected tool. No scenario change. |
| retail-4 | Add a scored assertion requiring explained changes and explicit confirmation before each order modification. Customer instructions and reference items remain unchanged. |
| retail-52 | Remove the impossible higher-zoom exchange preserving 24MP and SD storage. Expect no database changes and an explanation that no eligible replacement exists. Customer instructions remain unchanged. |
| airline-39 | Make the cancellation reason explicitly a change of plans and remove insured economy reservation MSJ4OA from expected cancellations. Retain the two eligible business cancellations. Insurance alone does not authorize cancellation for this reason. |

The three task replacements live in `src/tau2_bench/task_corrections.json`; benchmark loading and evaluation apply them outside the optimizer's editable resources. The retail tool fix is in `vendor/tau2/src/tau2/domains/retail/tools.py`. Original downloaded data and checksums are unchanged.

The uploaded JSONL rows reference task IDs. Therefore the corrected task file and tool must also be present in the integration source used by a hosted run. Uploading task-reference rows alone does not apply the fixes. The dataset bundle includes the corrected task file and fix manifest for provenance. A new integration build is required.

This is a corrected benchmark variant, not an unmodified public benchmark. Reevaluate both the baseline and optimized agent with the same fixes before comparing scores. Historical run flags and scores are unchanged.
