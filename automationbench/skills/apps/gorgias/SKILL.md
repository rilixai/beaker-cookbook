---
name: gorgias
description: Procedures for Gorgias ticket workflows (listing, tag and status eligibility, replies, and downstream processing).
---

# Scope Gorgias ticket batches

Use this procedure when a Gorgias ticket listing feeds customer replies or actions in other apps.

1. Treat the returned tickets as candidates, not an already-filtered work queue. Read their actual status and tags alongside the request and retrieved workflow policy. Mentioning a tag or status in `search_tools` does not filter a later `gorgias_get_tickets` call. Check returned count and pagination signals before treating the collection as complete.
2. Establish the eligible ticket IDs before the first write. Match required tags exactly; a similar subject or customer message does not replace a required tag. Keep requested or pending approval distinct from approved intake: satisfying an order's amount, category, or date rules does not itself authorize an approved-only ticket workflow. Derive the applicable gate from the request, observed workflow stages, and current policy; do not invent a universal approval tag or open-only rule. Apply explicit policy exclusions and overrides rather than assuming every exclusion always wins.
3. Branch only within that eligible set. An eligible ticket with missing information may still require a question, a not-found outcome, a denial, or an escalation. Do not apply those fallback responses to tickets outside the requested batch. Eligibility for processing and eligibility for the requested benefit are separate decisions.
4. Carry the same eligible ticket IDs through Gorgias replies, Gmail drafts, downstream issues, logs, and summaries. Before dispatch, check each planned action against the ticket's status, tags, and applicable policy decision. Do not add a helpful reply, triage note, rejection notice, or log entry to an out-of-scope ticket unless the request or policy explicitly requires that action in that destination. Reuse complete evidence already retrieved; this check need not add business-system calls.
