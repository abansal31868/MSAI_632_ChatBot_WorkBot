"""
Deterministic intent router for WorkBot's Task Automation / API & Tool
Integration features, plus the explicit "remember that ..." trigger for
long-term memory (Feature #6, see memory.py).

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

Multi-turn calendar slot-filling: a calendar request only has one hard
requirement to actually create an event -- a real date/time. If that's
missing, route_task asks for it and returns a "pending_task" dict instead
of executing immediately. The caller (app.py) is responsible for storing
that dict in session state and passing it back in on the *next* call, so
follow-up messages ("it's at 2pm", "with Alpna and Tim", "it's about
budget planning") get merged into the same in-progress request instead of
being evaluated as brand-new, context-free messages.
"""

import re

from calendar_integration import try_create_google_calendar_event
from memory import remember_fact
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
    losing genuine matches. Also requires >=3 chars to filter out spurious
    single-character matches.

    dateparser's search sometimes returns the date and time as two separate
    matches (e.g. "August 11" and "2 pm" from "on August 11 at 2 pm"), which
    would silently drop the time if we only kept the first hit. This merges
    matches that are close together in the original text (a small gap like
    " at " or ", " between them) into one combined phrase.
    """
    if not _HAS_DATEPARSER_SEARCH:
        return ""
    try:
        found = search_dates(text, settings={"PREFER_DATES_FROM": "future"})
    except Exception:
        return ""
    if not found:
        return ""

    candidates = []
    for matched_text, _ in found:
        matched_text = matched_text.strip()
        if len(matched_text) >= 3 and any(c.isalpha() for c in matched_text):
            idx = text.find(matched_text)
            if idx != -1:
                candidates.append((idx, idx + len(matched_text)))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: c[0])

    # Merge the first run of candidates that are close together in the
    # original text (small gaps like " at " between date and time).
    merged_start, merged_end = candidates[0]
    for start, end in candidates[1:]:
        if len(text[merged_end:start]) <= 6:
            merged_end = end
        else:
            break
    merged_text = text[merged_start:merged_end].strip()

    # Pull in a leading "next"/"this"/"coming" if dateparser left it out
    # (e.g. matched "Monday" from "next Monday"), and drop a leading
    # "on"/"at"/"for" left over from how the sentence was phrased.
    expanded = re.search(
        r"\b(next|this|coming)\s+" + re.escape(merged_text), text, re.IGNORECASE
    )
    result = expanded.group(0) if expanded else merged_text
    result = re.sub(r"^(on|at|for)\s+", "", result, flags=re.IGNORECASE)
    return result.strip()


def _extract_who(text: str) -> str:
    """Pull out who a meeting is with/attendees are, e.g. 'a call with
    Sarah' -> 'Sarah', or 'the attendees are Alpna and Tim' -> 'Alpna and Tim'."""
    match = re.search(r"\bwith\s+(.+)", text, re.IGNORECASE)
    if not match:
        match = re.search(
            r"\battendees?(?:\s+(?:are|is|include))?\s*:?\s*(.+)", text, re.IGNORECASE
        )
    if not match:
        return ""
    who = match.group(1).strip(" ,.")
    who = re.sub(r"\s+(?:for|at|on)\s*$", "", who, flags=re.IGNORECASE).strip(" ,.")
    return who


def _extract_topic(text: str) -> str:
    """Pull out what a meeting is about, e.g. 'it's about budget planning'
    -> 'budget planning'."""
    match = re.search(r"\b(?:it'?s|it is|its)?\s*about\s+(.+)", text, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip(" ,.")


def _compute_calendar_title(slots: dict) -> str:
    who, topic = slots.get("who", ""), slots.get("topic", "")
    if topic and who:
        return f"Meeting with {who} about {topic}"
    if topic:
        return f"{topic[:1].upper()}{topic[1:]} meeting"
    if who:
        return f"Meeting with {who}"
    return "Meeting"


def _finalize_calendar_event(slots: dict) -> str:
    title = _compute_calendar_title(slots)
    when = slots["when"]
    real_result = try_create_google_calendar_event(title=title, when_text=when)
    if real_result is not None:
        return real_result
    return create_calendar_event(title=title, when=when)


# --- Intent patterns -------------------------------------------------------
# Order matters: more specific patterns should come before more general ones.

_TODO_PATTERNS = [
    re.compile(r"remind me to (.+)", re.IGNORECASE),
    re.compile(r"add (?:a |an )?(?:to-?do|task|reminder)(?: to| for| that)?\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:log|create) (?:a )?(?:to-?do|task)(?: to| for)?\s+(.+)", re.IGNORECASE),
]

# Long-term memory (Feature #6): only fires on an explicit "remember
# that ..." / "please remember ..." request, never inferred from ordinary
# conversation -- see memory.py's docstring for why. Deliberately a
# different trigger word ("remember") from the to-do patterns above
# ("remind me to"), though "please remember to submit the report" will
# still match this block rather than _TODO_PATTERNS, storing "to submit
# the report" as a fact rather than a to-do item. That's a known rough
# edge in the phrasing overlap between "remember" and "remind", not a bug
# -- either intent produces a reasonable, visible result for the user.
_REMEMBER_PATTERNS = [
    re.compile(r"remember that (.+)", re.IGNORECASE),
    re.compile(r"please remember (.+)", re.IGNORECASE),
]

# Calendar trigger verbs need to be followed by an explicit calendar-ish
# noun (invite/meeting/event/etc.) -- otherwise generic verbs like "create"
# or "make" would hijack unrelated requests ("create a summary of...").
_CALENDAR_TRIGGER = re.compile(
    r"\b(?:schedule|set up|set-up|book|create|make|add)\b\s+(?:a |an |the )?"
    r"(?:calendar\s+(?:invite|event)|meeting|call|event|invite|appointment|"
    r"sync|1:1|1-on-1|standup|check-in|catch-up)s?\b\s*(.*)",
    re.IGNORECASE,
)

_EMAIL_PATTERN_WITH_RECIPIENT = re.compile(
    r"(?:draft|write|prepare|compose) (?:an |a )?email to (.+?)\s+"
    r"(?:about|regarding|for|re:?)\s+(.+)",
    re.IGNORECASE,
)
_EMAIL_PATTERN_NO_RECIPIENT = re.compile(
    r"(?:draft|write|prepare|compose) (?:an |a )?email\s+"
    r"(?:about|regarding|for|re:?)\s+(.+)",
    re.IGNORECASE,
)

_CANCEL_PATTERN = re.compile(r"^\s*(cancel|never\s*mind|forget it|nvm)\s*[.!]?\s*$", re.IGNORECASE)


def _looks_like_new_intent(text: str) -> bool:
    """Cheap check for whether a message clearly starts a different,
    fully-formed request, used to decide whether to abandon a pending
    calendar task rather than treat this message as continuing it."""
    if any(p.search(text) for p in _TODO_PATTERNS):
        return True
    if any(p.search(text) for p in _REMEMBER_PATTERNS):
        return True
    if _EMAIL_PATTERN_WITH_RECIPIENT.search(text) or _EMAIL_PATTERN_NO_RECIPIENT.search(text):
        return True
    if _CALENDAR_TRIGGER.search(text):
        return True
    return False


def _continue_pending_calendar(text: str, pending: dict) -> tuple[str, dict | None]:
    if _CANCEL_PATTERN.match(text):
        return "No problem, I've dropped that calendar request.", None

    when = _extract_datetime_phrase(text)
    who = _extract_who(text)
    topic = _extract_topic(text)

    if when:
        pending["when"] = when
    if who:
        pending["who"] = who
    if topic:
        pending["topic"] = topic

    if pending.get("when"):
        return _finalize_calendar_event(pending), None

    # Still missing the one hard requirement -- ask again, more specifically.
    title = _compute_calendar_title(pending)
    return f'Got it. What date and time should I put for "{title}"?', pending


def route_task(
    user_input: str, llm=None, pending_task: dict | None = None
) -> tuple[str | None, dict | None]:
    """Check user_input against known task intents.

    Returns (response, updated_pending_task):
    - response is a string if a tool ran or a clarifying question is being
      asked, or None if this message isn't a task at all (caller should
      fall through to the RAG chain).
    - updated_pending_task is a dict to store in session state and pass
      back in on the next call if a calendar request is still waiting on
      more info, or None if there's nothing pending anymore.

    `llm` is optional and only used by draft_email to write a real body
    instead of a bare template; every other tool ignores it.
    """

    text = user_input.strip()

    # --- Continue an in-progress calendar request, if one exists and this
    # message doesn't clearly start something else entirely. ---
    if pending_task is not None and pending_task.get("intent") == "calendar":
        if not _looks_like_new_intent(text):
            return _continue_pending_calendar(text, pending_task)
        # else: message looks like a fresh, complete request of its own --
        # fall through and abandon the pending one.

    # --- Remember (long-term memory, Feature #6) ---
    for pattern in _REMEMBER_PATTERNS:
        match = pattern.search(text)
        if match:
            fact = match.group(1).strip(" ?.!")
            return remember_fact(fact), None

    # --- To-do ---
    # We deliberately do NOT try to strip the date phrase out of the task
    # text -- that string-surgery caused mangled results like "update the
    # wiki by next" (from "...by next Wednesday"). A task that redundantly
    # repeats its due date is a much smaller problem than a broken one.
    for pattern in _TODO_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = match.group(1).strip(" ?.!")
            due_date = _extract_datetime_phrase(raw)
            return log_todo(task=raw, due_date=due_date), None

    # --- Calendar ---
    match = _CALENDAR_TRIGGER.search(text)
    if match:
        raw = match.group(1).strip(" ?.!")
        when = _extract_datetime_phrase(raw)
        # Extract who/topic from the remainder with the date phrase removed,
        # not from raw directly -- otherwise "with the design team for next
        # Monday at 10am" swallows the date into the "who" text.
        remainder = raw.replace(when, "", 1).strip(" ,.") if when else raw
        slots = {
            "intent": "calendar",
            "when": when,
            "who": _extract_who(remainder),
            "topic": _extract_topic(remainder),
        }
        if slots["when"]:
            return _finalize_calendar_event(slots), None
        # Missing the one hard requirement -- ask instead of guessing or
        # silently creating a placeholder "time not specified" event.
        title = _compute_calendar_title(slots)
        return f'I can set that up. What date and time should I put for "{title}"?', slots

    # --- Email ---
    match = _EMAIL_PATTERN_WITH_RECIPIENT.search(text)
    if match:
        recipient, key_points = match.group(1).strip(), match.group(2).strip()
        subject = key_points if len(key_points) < 60 else key_points[:57] + "..."
        return draft_email(recipient=recipient, subject=subject, key_points=key_points, llm=llm), None

    match = _EMAIL_PATTERN_NO_RECIPIENT.search(text)
    if match:
        key_points = match.group(1).strip()
        subject = key_points if len(key_points) < 60 else key_points[:57] + "..."
        result = draft_email(recipient="[recipient]", subject=subject, key_points=key_points, llm=llm)
        result += "\n\n_(No recipient was specified, so I used a placeholder -- let me know who this should go to and I'll redo it.)_"
        return result, None

    return None, None
