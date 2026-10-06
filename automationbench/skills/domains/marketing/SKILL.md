---
name: marketing
description: Procedures and playbooks for marketing-domain tasks (CRM contact cleanup/audits, influencer outreach, conversion tracking, social engagement, referrals, ad account audits).
---

Read general/workflow first. Marketing tasks carry tracking codes and SOPs sent by email or Slack; the newest one governs and its code must appear in your outputs.

## Contact data audits / cleanup
- Find ALL cleanup policy versions (Gmail, Slack, Drive) and use the most recent by date; its tracking code and its tag values replace older ones.
- Apply validation rules exactly (what to flag, what not to flag, system/do-not-flag notes, special sections such as legacy imports).
- Set the audit tag value from the newest policy on every contact the policy says to tag.
- Report: tracking code in subject and body, issues by category with the offending emails verbatim, no valid contacts listed, escalation note if the threshold is crossed.

## Influencer / partner outreach
- Read campaign guidance from the campaign manager (Gmail AND Slack channels). Include its campaign code and product name in every outreach email.
- Contact only rows not yet contacted, plus explicit re-contact instructions; skip rows with exclusivity/competitor notes or that guidance says to skip.
- Mention the person's platform and follower count verbatim; update the tracker row (contacted flag + note) for each email sent.

## Conversion tracking (ads)
- Closed deals may be announced in Gmail or sheets rather than in the CRM; if the CRM is empty, search Gmail.
- Apply the conversion policy (excluded accounts, test deals, minimum value, date window) and send one conversion per qualifying deal with its gclid and exact value.

## Social engagement
- Locate the SOP (often a Gmail message) before any action. Apply its action matrix per mention type (praise, question, complaint, competitor, spam).
- Complaints/outages usually go to an internal Slack alert channel with the handle and issue details instead of a public reply. Do not like/reply/retweet where the SOP does not call for it.

## Referrals / lead intake
- Skip referees who are already customers/contacts; use the latest tracking code; honor special-handling notes on existing contact properties; mark processed rows.

## Ad account audits
- Follow the paid-media policy over the ad-hoc request (e.g. pause instead of delete if policy forbids deletion); round computed metrics as told; name each affected campaign in the summary.
