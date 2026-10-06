---
name: intercom
description: Procedures for the Intercom app (contacts, conversations, tickets, tags, notes, companies, etc.).
---

- intercom_list_companies first to see existing companies (dedupe by name, case-insensitive).
- intercom_find_or_create_company(name, company_id=<source system ID>, website=<domain>) returns created true/false and the Intercom id; use it rather than intercom_create_company when a company may already exist.
- intercom_tag_company(company_id=<Intercom id from the response, not the source ID>, tag=<mapped tag>) once per mapped tag.
- Apply the tag mapping table exactly; source tags with no mapping row get no Intercom tag but the company is still synced.
