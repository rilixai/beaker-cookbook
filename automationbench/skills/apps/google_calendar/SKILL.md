---
name: google_calendar
description: Procedures for Google Calendar bookings, event titles, times, durations, and attendees.
---

When a task or policy specifies an event title, time, or duration, discover and use `google_calendar_create_detailed_event`. Put the required title in `summary`, the calendar ID in `calendarid`, and the start and end in `start__dateTime` and `end__dateTime`. Compute the end from the selected start and required duration. Keep dates and durations out of the title unless the naming rule includes them. Quick-add text can remain in the title without setting the intended times.

Use the attendee list allowed by the task and current policy. Keep a delegated invitation step delegated when the policy requires it. Compare the returned title, start, end, and attendees with the planned event before sending a booking confirmation or marking the request scheduled. A success flag alone does not confirm that those fields match. If an event was created with wrong fields, correct that event using its returned ID instead of creating another booking.
