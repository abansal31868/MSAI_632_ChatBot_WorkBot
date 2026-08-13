"""
Real Google Calendar integration for WorkBot's Task Automation feature.

This is a demo-facing upgrade over the local .ics file: when it works, the
event appears on the presenter's actual Google Calendar, which is a much
more convincing live demo than "trust me, it wrote to a file." When it
doesn't work (auth not set up yet, no network, bad time parse, quota
issue), every function here returns None so the caller (tools.py) can fall
back to the local .ics event instead of breaking the conversation.

One-time setup required before this will do anything (see README):
1. Enable the Google Calendar API in the same Cloud project used for Drive.
2. Create an OAuth 2.0 Client ID (Desktop app), save as oauth_credentials.json.
3. Add your Google account as a Test User on the OAuth consent screen.

First call opens a browser for one-time consent; after that a cached
token.json is reused automatically.
"""

import logging
import os
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

try:
    import dateparser
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
OAUTH_CLIENT_SECRET_PATH = "oauth_credentials.json"
TOKEN_PATH = "calendar_token.json"
DEFAULT_EVENT_DURATION_MINUTES = 30


def _get_calendar_credentials():
    """Load cached OAuth credentials, refreshing or running the one-time
    consent flow as needed. Returns None if anything is missing/fails."""
    if not _DEPS_AVAILABLE:
        return None
    if not os.path.exists(OAUTH_CLIENT_SECRET_PATH):
        return None

    creds = None
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception:
            _logger.exception("Failed to load cached Calendar credentials from %s", TOKEN_PATH)
            creds = None

    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    OAUTH_CLIENT_SECRET_PATH, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())
        except Exception:
            _logger.exception("Failed to obtain/refresh Calendar credentials")
            return None

    return creds


def try_create_google_calendar_event(title: str, when_text: str) -> str | None:
    """Attempt to create a real event on the user's primary Google Calendar.

    Returns a confirmation string (including a link to the event) on
    success, or None if this couldn't be done for any reason -- missing
    setup, auth failure, unparseable time, or an API error. Callers should
    treat None as "fall back to the local calendar file," not as an error
    to surface to the user.
    """
    if not _DEPS_AVAILABLE:
        return None

    if not when_text or when_text.strip().lower() == "time not specified":
        # No real time to schedule against -- can't create a timed calendar
        # event from this, let the caller use the local fallback instead.
        return None

    start_dt = dateparser.parse(when_text, settings={"PREFER_DATES_FROM": "future"})
    if start_dt is None:
        return None

    # dateparser returns a naive datetime (no UTC offset attached) for a
    # phrase like "August 12 at 2pm" -- there's no timezone info in the
    # text to parse. The Calendar API rejects a dateTime with no offset
    # and no explicit timeZone field ("Missing time zone definition for
    # start time."), so attach the machine's local timezone before this
    # goes anywhere near isoformat(). Since when_text is always meant in
    # whatever timezone the person typing it is in, the local system
    # timezone is the right interpretation here.
    if start_dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        start_dt = start_dt.replace(tzinfo=local_tz)

    end_dt = start_dt + timedelta(minutes=DEFAULT_EVENT_DURATION_MINUTES)

    creds = _get_calendar_credentials()
    if creds is None:
        return None

    try:
        service = build("calendar", "v3", credentials=creds)
        event_body = {
            "summary": title.strip(),
            "start": {"dateTime": start_dt.isoformat()},
            "end": {"dateTime": end_dt.isoformat()},
        }
        created_event = (
            service.events().insert(calendarId="primary", body=event_body).execute()
        )
    except Exception:
        _logger.exception("Google Calendar event creation failed, falling back to local .ics")
        return None

    link = created_event.get("htmlLink", "")
    # Always include the year, even for near-term dates: a bare month/day
    # phrase like "August 12" that's already passed this year gets rolled
    # to next year by PREFER_DATES_FROM="future" above, which is the
    # correct interpretation (you can't schedule a meeting in the past) --
    # but omitting the year from the confirmation made that silent, so an
    # event a full year out looked identical to one next week. Found via a
    # real case: "Meeting on August 12" asked for on August 13 landed on
    # August 12 *2027*, and nothing in the chat reply said so.
    when_display = start_dt.strftime("%A, %B %d, %Y at %I:%M %p")
    if link:
        return (
            f"Created calendar event: **{title.strip()}** ({when_display}) "
            f"— view it at {link}"
        )
    return f"Created calendar event: **{title.strip()}** ({when_display})"
