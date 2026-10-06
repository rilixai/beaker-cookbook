---
name: google_calendar
description: Procedures for the Google Calendar app (find events, create events, avoid conflicts).
---

- Use the calendar ID the task names (google_calendar_find_calendars lists them; "primary" may not be the right one).
- google_calendar_find_event(calendarid, start_time, end_time) to list existing events; never double-book a person who already has an event, and do not overlap new events with each other.
- google_calendar_create_detailed_event(calendarid, summary, start__dateTime, end__dateTime ISO with offset, attendees, description). Summary must contain the exact names/subjects the task asks for (e.g. "<Full Name> - <Subject>").
- Schedule in business hours on dates after the task's "today".
