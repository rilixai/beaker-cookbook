---
name: wave
description: Procedures for Wave accounting (customers, invoices, products, sales).
---

# Wave

- `wave_find_customer(name=..., email=...)` before creating a customer; reuse the
  existing customer id.
- `wave_create_invoice(customer__id, product_id, description, price, quantity,
  invoice_date, due_date)`: the invoice total is `price x quantity`. Bill one invoice
  per client/project as the task defines; use `wave_find_product` / `wave_create_product`
  for the line item.
- `wave_send_invoice` only sets the invoice status to SENT; it does not deliver an
  email. When the task says to send the invoice to / notify the client, also
  `gmail_send_email` the client with the invoice total written with comma thousands
  separators.
- `wave_list_invoices(customer_id=None, status=None)` lists all invoices; use it to
  verify what you created.
