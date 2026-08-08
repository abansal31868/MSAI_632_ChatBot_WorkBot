"""
Deterministic intent router for WorkBot's Task Automation / API & Tool
Integration features.

Why deterministic instead of LLM tool-calling: see test_tool_calling.py and
the README. Short version: measured ~56% tool-selection accuracy on the
free-tier model, plus the free Hugging Face Inference Providers credit pool
ran out mid-test (402 Payment Required). A keyword/regex router costs zero
extra model calls and is 100% consistent, at the cost of only handling
phrasings we've explicitly matched.

This is intent classification with slot filling -- match the utterance to a
known intent, extract its parameters, call the matching function in
tools.py. If nothing matches, route_task returns None and app.py falls
through to the normal RAG chain.
"""

import re

from tools import create_calendar_event, draft_email, log_todo

try:
    from dateparser.search import search_dates
    _HAS_DATEPARSER_SEARCH = True
except ImportError:
    _HAS_DATEPARSER_SEARCH = False


def _extract_datetime_phrase(text: str) -> str:
    """Best-effort extraction of a date/time phrase from free text.
    Falls back to returning an empty string if dateparser isn't installed
    or nothing looks like a date.

    Skips any match with no alphabetic characters at all (e.g. dateparser
    misreading "1:1" as a clock time) -- real date/time phrases almost
    always include a word (a weekday, "tomorrow", "am"/"pm", a month), so
    this filters out the most common false positives without much risk of
    losing genuine matches.
    """
    if not _HAS_DATEPARSER_SEARCH:
        return ""
    try:
        found = search_dates(text, settings={"PREFER_DATES_FROM": "future"})
    except Exception:
        return ""
    if not found:
        return ""
    for matched_text, _ in found:
        matched_text = matched_text.strip()
        # Require >=3 chars and at least one letter -- filters out spurious
        # single-character/short matches while still allowing "2pm", "9am".
        if len(matched_text) >= 3 and any(c.isalpha() for c in matched_text):
            # dateparser sometimes leaves a leading "next"/"this"/"coming"
            # out of the match (e.g. matches "Monday" from "next Monday");
            # pull it back in if present so we don't leave a dangling
            # "for next" behind in the remaining text.
            expanded = re.search(
                r"\b(next|this|coming)\s+" + re.escape(matched_text),
                text,
                re.IGNORECASE,
            )
            return expanded.group(0).strip() if expanded else matched_text
    return ""


def _extract_who(remainder: str) -> str:
    """Pull out who a meeting is with from whatever's left after the date
    phrase has been removed, e.g. 'a call with Sarah' -> 'Sarah'."""
    match = re.search(r"\bwith\s+(.+)", remainder, re.IGNORECASE)
    if not match:
        return ""
    who = match.group(1).strip(" ,.")
    who = re.sub(r"\s+(?:for|at|on)\s*$", "", who, flags=re.IGNORECASE).strip(" ,.")
    return who


# --- Intent patterns -------------------------------------------------------
# Order matters: more specific patterns should come before more general ones.

_TODO_PATTERNS = [
    re.compile(r"remind me to (.+)", re.IGNORECASE),
    re.compile(r"add (?:a |an )?(?:to-?do|task|reminder)(?: to| for| that)?\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:log|create) (?:a )?(?:to-?do|task)(?: to| for)?\s+(.+)", re.IGNORECASE),
]

_CALENDAR_TRIGGER = re.compile(r"\b(?:schedule|set up|set-up|book)\b\s+(.+)", re.IGNORECASE)

_EMAIL_PATTERNS = [
    re.compile(r"draft (?:an )?email to (.+?)\s+(?:about|regarding|re:?)\s+(.+)", re.IGNORECASE),
    re.compile(r"write (?:an )?email to (.+?)\s+(?:about|regarding|re:?)\s+(.+)", re.IGNORECASE),
]


def route_task(user_input: str, llm=None) -> str | None:
    """Check user_input against known task intents. Returns a confirmation
    string if a task was executed, or None if this isn't a task request
    (caller should fall through to the RAG chain).

    `llm` is optional and only used by draft_email to write a real body
    instead of a bare template; every other tool is fully deterministic
    and ignores it.
    """

    text = user_input.strip()

    # --- To-do ---
    # We deliberately do NOT try to strip the date phrase out of the task
    # text -- that string-surgery is what caused mangled results like
    # "update the wiki by next" (from "...by next Wednesday"). A task that
    # redundantly repeats its due date ("update the wiki by next Wednesday",
    # due "Wednesday") is a much smaller problem than a grammatically
    # broken one.
    for pattern in _TODO_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(1).strip(" ?.!")
            due_date = _extract_datetime_phrase(raw)
            return log_todo(task=raw, due_date=due_date)

    # --- Calendar ---
    # Find the date/time phrase first and independently, then look for a
    # "with <who>" clause in whatever's left. Doing these two extractions
    # separately (instead of one combined regex) avoids the greedy-match
    # bugs where "with Sarah tomorrow at 2pm" swallowed "tomorrow" into the
    # "who" group, or "sync"/"1:1" weren't recognized as meeting words at
    # all.
    match = _CALENDAR_TRIGGER.search(text)
    if match:
        raw = match.group(1).strip(" ?.!")
        when = _extract_datetime_phrase(raw)
        remainder = raw.replace(when, "", 1).strip(" ,.") if when else raw
        who = _extract_who(remainder)
        title = f"Meeting with {who}" if who else "Meeting"
        when = when or "time not specified"
        return create_calendar_event(title=title, when=when)

    # --- Email ---
    for pattern in _EMAIL_PATTERNS:
        match = pattern.search(text)
        if match:
            recipient, key_points = match.group(1).strip(), match.group(2).strip()
            subject = key_points if len(key_points) < 60 else key_points[:57] + "..."
            return draft_email(recipient=recipient, subject=subject, key_points=key_points, llm=llm)

    return None
