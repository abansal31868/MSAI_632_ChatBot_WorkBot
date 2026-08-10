"""
Google Calendar integration for WorkBot -- PLACEHOLDER.

router.py imports try_create_google_calendar_event from this module. This
file didn't exist in the repo yet (confirmed missing via `find . -name
"calendar_integration.py"`), so this is a minimal stub to unblock
importing router.py and running the app, NOT a real integration.

try_create_google_calendar_event() always returns None here, which
router.py already handles as "no real calendar available" -- see
_finalize_calendar_event() in router.py, which falls back to
tools.create_calendar_event() (the local .ics file) whenever this
returns None. So calendar requests still work end-to-end right now, they
just land in workbot_calendar.ics instead of an actual Google Calendar.

TODO (whoever picks this up): implement a real version using the same
service-account credentials.json pattern ingest.py already uses, calling
the Calendar API's events().insert() on the user's calendar. Return the
real Calendar API response's confirmation text (e.g. including the event
link) on success, and return None on any failure/exception so the
tools.py fallback keeps working -- don't let a Calendar API error crash
the chat.
"""


def try_create_google_calendar_event(title: str, when_text: str) -> str | None:
    """Attempt to create a real Google Calendar event. Returns None
    (not implemented yet) so router.py falls back to the local .ics file
    in tools.py instead of failing."""
    return None
