---
name: support
description: Procedures and playbooks for support-domain tasks (tickets, SLAs, escalations, etc.).
---

This is the workflow playbook for support-domain tasks (tickets, SLAs, escalations, etc.).

When asked to reply to every ticket in a scope, keep each ticket in the response list even if it lacks an identifier needed for lookup. Ask the customer for the missing order or record number in that ticket, or use the prescribed escalation. Keep explicit policy skips outside this response list.

Track response completion separately from lookup success. If the task limits a log to successfully looked-up records, write a log row only after an exact record match. A reply, a missing-number request, or an escalation for an unknown number does not make the lookup successful. Before the summary, check that each in-scope ticket has its required response and that every proposed log row has a qualifying lookup.
