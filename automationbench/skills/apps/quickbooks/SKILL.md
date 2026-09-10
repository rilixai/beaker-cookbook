---
name: quickbooks
description: Procedures for QuickBooks Online (customers, vendors, invoices, bills, payments, query).
---

# QuickBooks Online

## Reading
- `quickbooks_query(query="SELECT * FROM <Entity>")` returns **every** record of that
  entity (Invoice, Bill, Payment, Customer, Vendor, Item, Estimate); WHERE clauses are
  ignored. Use it once per entity to get the full list, then filter yourself. Prefer it
  over repeated `quickbooks_find_*` calls.
- `quickbooks_find_customer(name=...)` is a substring match on display name;
  `where_clause="DisplayName = '...'"` is exact. Always check for an existing customer /
  vendor before creating one (also under a shorter or differently punctuated name).
- Pull the full payment/invoice list before a reconciliation and compare it against the
  other system in full; don't stop at the first few references.

## Writing
- One invoice per customer per billing run; put the computed total in `line_amount`
  (plain number), the customer's exact name in `customer`/`customer_name`, and a
  `line_description` naming the period/project.
- `quickbooks_send_invoice` only marks the invoice sent; it does not deliver an email.
  When the task asks to notify or email the customer, send the details (including the
  total) with `gmail_send_email`.
- Do not create records for entities the workspace policy says to hold, verify first,
  or skip; do not email their contacts either.
- Re-read the created records (query the entity again) before finishing.
