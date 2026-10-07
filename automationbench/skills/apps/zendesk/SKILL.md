---
name: zendesk
description: Procedures for Zendesk, including organization consolidation and ticket reassignment.
---

When consolidating Zendesk organizations, read the source-to-target plan and any do-not-merge rules before making changes. Apply exclusions to both the organizations and their linked tickets. An archive tag or a target note alone does not move tickets.

If no organization merge action is available, use `zendesk_get_tickets` to find tickets whose `organization_id` matches each allowed source. Use each returned ticket's `id` with `zendesk_update_ticket`, setting `organization_id` to that source's target ID. Include all ticket statuses unless the governing rules limit them. Change only the organization link; use null for other optional update fields. Leave tickets already at the target or outside the allowed sources unchanged. Check that each update result shows the intended target before recording that merge as complete.

Then apply the requested source archive tags, target notes, result records, and summary. Preserve existing tags and notes when adding to them. A source with no tickets can still receive the requested archive tag and result record. Count completed source-to-target pairs separately from moved tickets.

When syncing Zendesk tickets to another app, treat each whitelist in the sync rules as an exact allowed set. Check each source ticket against every whitelist and tag exclusion before creating a destination ticket or marking the source as synced. A status outside its whitelist is ineligible even if the ticket is unresolved. Count only eligible tickets whose destination creation succeeded.
