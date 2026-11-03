---
name: bamboohr
description: Procedures for BambooHR employee lookup and profile creation.
---

When asked to create a BambooHR employee profile, use `bamboohr_employeeCreate` with the person's source fields. Discover that exact tool before writing.

`bamboohr_employeeSearch` only searches, despite its "Find or Create Employee" description. Supplying profile fields to it does not create an employee. If a lookup returns no match and the task calls for a new profile, call `bamboohr_employeeCreate` instead of repeating the search with creation fields. Reuse a matching employee already found; do not create a duplicate. Confirm that the create result contains the new employee before treating the profile step as complete.
