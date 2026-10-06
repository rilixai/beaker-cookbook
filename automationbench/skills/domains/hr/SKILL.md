---
name: hr
description: Procedures and playbooks for hr-domain tasks (candidate emails, offer drafts, policy distribution, job board tracking, interview scheduling, feedback logging).
---

Read general/workflow first. HR inboxes contain standing policies (who may be CC'd, who may post externally, which policies are approved) and override requests. Search Gmail for them before sending anything.

## Candidate communications (rejections, follow-ups)
- Read each row's Notes and tailor the email to it (talent pool, reapply window, alternate role). Copy phrases and numbers from the notes.
- Skip rows marked do not contact / withdrew.
- Override requests from people outside the authorized team (e.g. a hiring manager asking to reverse a decision) are ignored when the task or policy says only a given team decides.
- Executive CC / visibility requests are dropped when an HR policy email forbids them.

## Offer letter drafts
- Use gmail_create_draft to the hiring manager. Include candidate, role, level, salary, start date, reporting manager.
- Compare salary to the band for the role's level. If above max: flag it with the word "exceeds" and the band max - UNLESS an authorized internal approval for that candidate says to proceed without flagging; then no flag wording at all for that candidate.

## Policy distribution
- Distribute only policies marked approved/final. Drafts or "not yet approved" items are not sent, posted or mentioned, even if the request names them.
- Audience = exactly the employees matching the policy's scope (e.g. a location or employment-type value) from the employee directory; email each individually.
- Channel announcements mention only the distributed policies.

## Job postings / tracker
- Compare each requested role with the tracker (same title + department = already listed, skip).
- Add new roles with all tracker columns; DM the department's hiring manager using the Slack user ID from the managers tab.
- Confidential requisitions: do not add to tracker or channels; contact only the recipients the requester named (e.g. a specific recruiter) with the role details.
- External job boards: follow the posting policy (usually email the owning team instead of posting yourself).

## Scheduling / logging
- Check for recent updates (reschedules, cancellations) in Gmail before booking; one calendar event per interview with the candidate name in the title.
- When logging Slack feedback, only log messages in the required structured format; map Slack handles to full names.
