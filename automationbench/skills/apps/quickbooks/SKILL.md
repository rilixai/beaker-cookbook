---
name: quickbooks
description: Procedures for the QuickBooks app (customers, vendors, invoices, bills).
---

- quickbooks_find_customer / find_vendor often return found:false; that does not block invoicing.
- quickbooks_create_invoice_by_name(name=<customer name>, line_amount, line_description, line_item_qty "1", line_item_price, txn_date, billing_email) creates the invoice; read TotalAmt and DocNumber from the response.
- Create exactly one invoice per customer per run; do not call both create_invoice and create_invoice_by_name for the same customer.
- line_amount / line_item_price are plain numbers (1234.5); client emails show the total as "$1,234.50".
