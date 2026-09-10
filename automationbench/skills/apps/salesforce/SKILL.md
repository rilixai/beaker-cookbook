---
name: salesforce
description: Procedures for the Salesforce app (records, queries, opportunities, updates, etc.).
---

# Salesforce

## Reading
- `salesforce_query(object_type="<Object>")` with no `where_clause` returns every record
  of that object. When a task says the data "is in the CRM", query in one parallel batch
  all objects that can describe an entity: `Account`, `Contact`, `Opportunity`, `Case`,
  `Lead`, `Task`, `Note`. Holds, flags, statuses and review requirements a policy refers
  to are often recorded on a related `Case` or `Note`, not on the Account.
- Join contacts to accounts through `account_id`; use the contact whose title/role
  matches the one the task names (billing, AP, primary) as the recipient.

## Writing
- `salesforce_update_record(object, recordId, fields={...})` for field changes;
  `salesforce_note_create` / `salesforce_case_comment_create` for annotations.
- Do not create downstream records (customers, invoices) or email contacts for an
  entity that a case, note or status marks as on hold / under review.
