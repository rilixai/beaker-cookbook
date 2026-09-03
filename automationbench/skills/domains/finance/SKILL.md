---
name: finance
description: Procedures and playbooks for finance-domain tasks (invoices, budgets, reporting, etc.).
---

# Finance execution workflow

1. Use `search_tools` only to discover application actions and their exact schemas; it does not search company records. If the request omits a source location, discover an appropriate file, worksheet, message, or business record with a provider search/list action. Never invent IDs or ask the user for data before trying the available business systems.
2. Read both the governing policy and the complete current transaction/entity set before calculating or acting. When the request mentions current targets, recent changes, status updates, or an exception, collect the observable date, status, approval, owner, and supersession signals and apply the authoritative task-specific source rather than a generic finance assumption. Follow paging/count signals and preserve exclusions.
3. Derive amounts, dates, entity sets, thresholds, and comparisons only from the request and returned records. Bind values from the same entity, keep source strings verbatim where requested, and format only computed outputs as instructed. Treat null, zero, unchanged, inactive, rejected, and out-of-period values distinctly.
4. Execute every requested artifact or state change with the discovered application action; prose in the final response is not a journal entry, transfer, sheet update, email, or message. Use returned stable IDs, put result-dependent calls in a later turn, avoid duplicate or unrequested writes, and preserve protective outcomes such as holds and no-action thresholds.
5. Inspect each structured result for actual success. Repair tool-name, parameter, or destination errors before continuing, then re-read critical state when possible. Finish only after all required writes and notifications have succeeded.
