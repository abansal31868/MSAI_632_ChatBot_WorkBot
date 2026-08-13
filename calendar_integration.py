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

import os
from datetime import timedelta

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
        return None

    link = created_event.get("htmlLink", "")
    when_display = start_dt.strftime("%A, %B %d at %I:%M %p")
    if link:
        return (
            f"Created calendar event: **{title.strip()}** ({when_display}) "
            f"— view it at {link}"
        )
    return f"Created calendar event: **{title.strip()}** ({when_display})"
