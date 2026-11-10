---
name: support
description: Procedures and playbooks for support-domain tasks (tickets, SLAs, escalations, etc.).
---

When syncing or migrating customer tickets between support apps, preserve the requester relationship unless the migration policy explicitly removes it.

1. Apply the ticket eligibility rules before creating destination records or contacts.
2. Read each eligible ticket's requester email. If the ticket only gives a requester or contact ID, look up that ID in the source app to retrieve the person's email and name. Reuse source details already retrieved.
3. Find the destination contact by exact email and reuse it. Create a contact from the retrieved source details only when none exists. Source and destination IDs belong to different apps; never copy a source requester ID into a destination requester field.
4. Link the destination ticket to that contact. Freshdesk's create-ticket tool can find or create the contact when you supply `email` and leave `requester_id` unset. Zoho Desk's create-ticket tool needs the destination `contact_id`, so resolve or create the contact first. Keep the workflow's other field mappings.
5. Check that the returned ticket has the destination requester or contact link before marking the source ticket migrated or synced. A successful ticket creation with an empty requester field does not preserve the customer relationship. If source identity is missing, seek it in the source app; do not invent an email or substitute a different person.
