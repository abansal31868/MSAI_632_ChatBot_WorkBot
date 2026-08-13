"""
Task Automation tools for WorkBot.

Each function here is a self-contained, local, free action: no paid APIs,
no external services. They're called directly by router.py's deterministic
dispatcher rather than through LLM tool-calling (see test_tool_calling.py
and the README Troubleshooting section for why: free-tier tool-calling
reliability came in around 56% in testing, plus Hugging Face's Inference
Providers free credits ran out mid-test, so keeping these deterministic
means Task Automation works every time and doesn't compete with the RAG
chat feature for the same limited quota).

Storage:
- To-dos: local SQLite table (workbot_data.db)
- Calendar events: local .ics file (workbot_calendar.ics), importable into
  any real calendar app
- Email drafts: local .txt files under email_drafts/
"""

import os
import re
import sqlite3
import uuid
from datetime import datetime

from email_integration import try_send_email
from personalization import get_profile

DB_PATH = "workbot_data.db"
CALENDAR_PATH = "workbot_calendar.ics"
DRAFTS_DIR = "email_drafts"


# --- Setup -------------------------------------------------------------

def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            due_date TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _ensure_calendar_file():
    if not os.path.exists(CALENDAR_PATH):
        with open(CALENDAR_PATH, "w") as f:
            f.write("BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//WorkBot//EN\nEND:VCALENDAR\n")


# --- Tools ---------------------------------------------------------------

def log_todo(task: str, due_date: str = "") -> str:
    """Add a to-do item to the local store."""
    conn = _get_db()
    conn.execute(
        "INSERT INTO todos (task, due_date, created_at) VALUES (?, ?, ?)",
        (task.strip(), due_date.strip(), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()

    if due_date:
        return f"Added to your to-do list: **{task.strip()}** (due {due_date.strip()})"
    return f"Added to your to-do list: **{task.strip()}**"


def create_calendar_event(title: str, when: str) -> str:
    """Create a calendar event and append it to the local .ics file."""
    _ensure_calendar_file()

    event_id = uuid.uuid4().hex
    now_stamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    vevent = (
        "BEGIN:VEVENT\n"
        f"UID:{event_id}@workbot\n"
        f"DTSTAMP:{now_stamp}\n"
        f"SUMMARY:{title.strip()}\n"
        f"DESCRIPTION:Created by WorkBot from a natural-language request. "
        f"Requested time: {when.strip()}\n"
        "END:VEVENT\n"
    )

    with open(CALENDAR_PATH, "r") as f:
        content = f.read()
    content = content.replace("END:VCALENDAR", vevent + "END:VCALENDAR")
    with open(CALENDAR_PATH, "w") as f:
        f.write(content)

    return f"Created calendar event: **{title.strip()}** ({when.strip()}) — saved to {CALENDAR_PATH}"


_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _clean_subject(raw_subject: str) -> str:
    """Turn a raw captured phrase like 'the budget review' into a proper
    subject line like 'Budget Review'."""
    cleaned = _LEADING_ARTICLES.sub("", raw_subject.strip())
    return cleaned[:1].upper() + cleaned[1:] if cleaned else raw_subject.strip()


def _draft_body_with_llm(llm, recipient: str, subject: str, key_points: str) -> str | None:
    """Try to generate a real email body with the chat model. Returns None
    on any failure so the caller can fall back to the template body --
    this keeps draft_email working even if the model/provider is down or
    the free quota is exhausted."""
    if llm is None:
        return None
    try:
        prompt = (
            "Write a brief, professional email body (3-5 sentences, no subject "
            "line, no salutation, no sign-off -- just the body paragraph(s)) "
            f"to {recipient.strip()} about: {key_points.strip()}. "
            "Keep it concise and workplace-appropriate. "
            "IMPORTANT: Do not invent specific details that were not provided "
            "-- no made-up dates, times, locations, room names, attachments, or "
            "attendee names. If the message would naturally include a detail "
            "like that, use an explicit bracketed placeholder instead, e.g. "
            "[date], [time], [location], so the user knows to fill it in "
            "before sending."
        )
        result = llm.invoke(prompt)
        content = getattr(result, "content", None) or str(result)
        return content.strip() or None
    except Exception:
        return None


def _compose_email(recipient: str, subject: str, key_points: str, llm=None) -> tuple[str, str, str]:
    """Shared composition logic for both drafting and sending: builds a
    subject, a body (LLM-written if possible, template otherwise), and the
    full plain-text draft. Returns (subject, body, full_draft_text)."""
    subject = _clean_subject(subject)

    body = _draft_body_with_llm(llm, recipient, subject, key_points)
    if body is None:
        body_lines = [p.strip() for p in re.split(r"[.;\n]", key_points) if p.strip()]
        body = "\n".join(f"- {line}" for line in body_lines) if body_lines else key_points.strip()

    # Personalization (Feature #5, personalization.py) already has a "Your
    # profile" sidebar form with a Name field -- reuse that here instead of
    # a bare "[Your name]" placeholder, so setting your name once in the
    # sidebar is enough for it to show up in every email sign-off too.
    sign_off_name = get_profile().get("name") or "[Your name]"

    draft = (
        f"To: {recipient.strip()}\n"
        f"Subject: {subject}\n\n"
        f"Hi {recipient.strip().split()[0] if recipient.strip() else 'there'},\n\n"
        f"{body}\n\n"
        f"Best,\n"
        f"{sign_off_name}\n"
    )
    return subject, body, draft


def _save_draft_file(recipient: str, draft: str) -> str:
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    filename = os.path.join(
        DRAFTS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{recipient.strip().split()[0] if recipient.strip() else 'draft'}.txt"
    )
    with open(filename, "w") as f:
        f.write(draft)
    return filename


def draft_email(recipient: str, subject: str, key_points: str, llm=None) -> str:
    """Draft an email (not sent) and save it locally as a .txt file.

    If `llm` is provided, ask it to write the body from key_points; if that
    fails for any reason (or llm is None), fall back to a simple bulleted
    template so this never breaks the demo.
    """
    _, _, draft = _compose_email(recipient, subject, key_points, llm=llm)
    filename = _save_draft_file(recipient, draft)
    return f"Drafted an email to **{recipient.strip()}** (saved to `{filename}`):\n\n```\n{draft}\n```"


def send_email(recipient: str, subject: str, key_points: str, llm=None) -> str:
    """Actually send an email via email_integration.try_send_email, with a
    local .txt draft always saved alongside it as a fallback/audit trail.

    Demo safety: the real send never goes to `recipient` -- see
    email_integration.py for why. If sending isn't configured (no Gmail
    App Password set up) or the send fails for any reason, this falls back
    to draft-only behavior instead of breaking the conversation.
    """
    subject, body, draft = _compose_email(recipient, subject, key_points, llm=llm)
    filename = _save_draft_file(recipient, draft)

    sent_confirmation = try_send_email(parsed_recipient=recipient, subject=subject, body=draft)
    if sent_confirmation is not None:
        return f"{sent_confirmation}\n\n(Also saved a local copy to `{filename}`.)"

    return (
        f"Email sending isn't set up yet, so I saved this as a draft instead "
        f"(`{filename}`) rather than losing it:\n\n```\n{draft}\n```\n\n"
        f"_(See the README's email setup step to enable real sending.)_"
    )
