---
name: finance
description: Procedures and playbooks for finance-domain tasks (invoices, budgets, reporting, etc.).
---

This is the workflow playbook for finance-domain tasks (invoices, budgets, reporting, etc.).

When a task asks you to email a customer the details of an invoice you created, compose and send the message with `gmail_send_email`. Include the source order reference when present and the created invoice's number, total, and due date. Use the accounting app's built-in invoice delivery instead only when the task explicitly requests that delivery method.
