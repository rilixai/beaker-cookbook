---
name: salesforce
description: Procedures for the Salesforce app (records, queries, opportunities, updates, etc.).
---

This is the operating procedure for the Salesforce app (records, queries, opportunities, updates, etc.).

When a policy makes an action depend on an account's credit or support status, check the Account and its related Cases before acting. Cases can carry credit holds and support issues that are absent from the Account record. A missing Account status field or a failed query does not show that the account is clear.

Use `salesforce_query` with object_type `Case` and where_clause `AccountId = '<returned Account ID>'`, then read the returned Status, Subject, and Description. Apply only the holds or exclusions named by the policy. Reuse related Cases already read.

When reconciling existing deals with Salesforce and the task forbids regressed deals, compare the source StageName with the stage already stored in the destination before any write. A source stage earlier in the sales pipeline is a regression even if the CRM record is otherwise current. Leave that whole deal unchanged; do not update its amount first or write a backward stage and repair it later. For equal or later stages, continue the permitted amount and stage updates, including currency conversion. Use the workflow's stage order when one is supplied.
