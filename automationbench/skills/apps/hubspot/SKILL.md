---
name: hubspot
description: Procedures for the HubSpot app (contacts, deals, custom properties).
---

- hubspot_get_all_contacts returns all contacts with core fields; custom properties (notes, source, audit tags) may sit under properties - read them for special-handling notes.
- Set custom properties with hubspot_update_contact(contact_id, additional_properties_json='{"prop_name":"value"}'); same argument on hubspot_create_contact.
- Before creating a contact, check for an existing one with the same email.
