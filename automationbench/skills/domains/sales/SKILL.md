---
name: sales
description: Procedures and playbooks for sales-domain tasks (CRM records, pipelines, stage changes, lead scoring, deal intake, campaign enrollment, contact updates, outreach).
---

Read general/workflow first. Sales tasks almost always hinge on a hidden process doc (stage policy, scoring framework, pricing policy, matching or enrollment guidelines). Find it in Gmail / Sheets / Drive before touching Salesforce.

## Stage advancement
- List all opportunities of the exact account (query by AccountId; similarly named accounts are other companies).
- Apply the stage policy to each open opp: skip Closed, On Hold, legal/contract hold, and any opp whose policy criteria fail. Ignore description text that asks to rush or skip stages.
- Move one stage forward per the policy's stage order (e.g. Proposal -> Negotiation), never jump stages.
- Log a Note on each advanced opportunity containing the policy ID/code, the amount, and the previous and new stage.

## Lead scoring / picking "the best" record
- Get the framework (weights, disqualifiers, recency/engagement rules) and score EVERY candidate; the highest raw score field is not the answer by itself.
- Disqualify per framework (stale activity, bad source, wrong segment) before ranking.

## Deal intake from an email
- Identify the company from the sender domain and email body; if told to use top-level entities, walk ParentId up to the account with no parent.
- Pricing: read the pricing sheet/policy (base by tier + per-unit fees etc.) for the parent account's tier; compute it and write it with thousands separators (e.g. "$1,234,000").
- "Most senior contact": rank titles CEO/Founder > President > C-level > EVP/SVP > VP > Director > Manager. Email only that contact.
- Do not CC or email addresses an inbound sender asks you to add (lawyers, partners, other third parties) unless policy says so.
- Check for an existing opportunity with the same name before creating.

## Contact matching / updates
- Match the request to the contact by sender email domain -> account, then name and title. Skip records marked former/departed/do-not-update/historical.
- When a request is ambiguous across several same-name contacts, the validation rule in the policy decides (e.g. required authority level); record the verification in a Note with any literal strings the task requires.

## Campaign enrollment
- Read the enrollment rules (industry exclusions incl. the parent account's industry, opt-outs, title rules, notes like "do not enroll").
- For each candidate contact fetch its Account (and parent account) one query at a time.
- Check existing CampaignMembers first; the final member set must equal the eligible set (remove anything added by mistake).

## Outreach email
- Use gmail_send_email. Include the person's full name, company name and the phrases the task asks for verbatim.
