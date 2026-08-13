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

Multi-turn slot-filling: a calendar request only has one hard requirement
to actually create an event -- a real date/time. An email request's hard
requirement is a topic (a recipient can fall back to a "[recipient]"
placeholder, but there's nothing to write about with no topic at all). If
either is missing, route_task asks for it and returns a "pending_task"
dict instead of executing immediately. The caller (app.py) is responsible
for storing that dict in session state and passing it back in on the
*next* call, so follow-up messages ("it's at 2pm", "with Alpna and Tim",
"inviting him to the design meeting today") get merged into the same
in-progress request instead of being evaluated as brand-new, context-free
messages.
"""

import re

from calendar_integration import try_create_google_calendar_event
from memory import remember_fact
from tools import create_calendar_event, draft_email, log_todo, send_email

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


# Both "who" and "topic" extraction stop at the same set of connector
# words/punctuation rather than capturing greedily to the end of the
# string -- an unbounded ".+" here means anything trailing the "with"
# clause (a topic phrase, "on my calendar", a whole second sentence like
# "Meeting is on August 12...") gets swallowed into the title. E.g. "Add a
# meeting invite with Tim from marketing on my calendar to discuss a
# marketing strategy for a new idea" used to produce who="Tim from
# marketing on my calendar to discuss a marketing strategy for a new
# idea" instead of stopping at "to discuss".
_WHO_STOP = r"(?=\s+\b(?:about|regarding|concerning|to\s+discuss|to\s+talk\s+about)\b|[.,!?]|$)"
_TOPIC_STOP = r"(?=\s+\bwith\b|[.!?]|$)"


def _extract_who(text: str) -> str:
    """Pull out who a meeting is with/attendees are, e.g. 'a call with
    Sarah' -> 'Sarah', or 'the attendees are Alpna and Tim' -> 'Alpna and Tim'.
    Stops at a topic connector, punctuation, or end of string rather than
    swallowing everything that follows."""
    match = re.search(r"\bwith\s+(.+?)" + _WHO_STOP, text, re.IGNORECASE)
    if not match:
        match = re.search(
            r"\battendees?(?:\s+(?:are|is|include))?\s*:?\s*(.+?)" + _WHO_STOP,
            text,
            re.IGNORECASE,
        )
    if not match:
        return ""
    who = match.group(1).strip(" ,.")
    who = re.sub(r"\s+(?:for|at|on)\s*$", "", who, flags=re.IGNORECASE).strip(" ,.")
    return who


def _extract_topic(text: str) -> str:
    """Pull out what a meeting is about, e.g. 'it's about budget planning'
    -> 'budget planning', or 'to discuss the roadmap' -> 'the roadmap'.
    Stops at a "with" clause, punctuation, or end of string.

    Same trailing-preposition trim as _extract_who: when the date phrase
    gets sliced out of the remainder upstream (e.g. "about the roadmap on
    August 20 at 9am" -> "about the roadmap on" once "August 20 at 9am" is
    removed), a bare trailing "on"/"at"/"for" is left dangling with no
    punctuation before it for the stop-lookahead to catch.
    """
    match = re.search(
        r"\b(?:it'?s|it is|its)?\s*(?:about|regarding|concerning|to\s+discuss|"
        r"to\s+talk\s+about)\s+(.+?)" + _TOPIC_STOP,
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    topic = match.group(1).strip(" ,.")
    topic = re.sub(r"\s+(?:for|at|on)\s*$", "", topic, flags=re.IGNORECASE).strip(" ,.")
    return topic


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


# A recipient phrase runs from "to" up to whichever comes first: a known
# topic-introducing word/phrase, sentence-ending punctuation, or the end of
# the string. This deliberately does NOT require "about"/"regarding" to
# immediately follow -- earlier versions did, which meant natural phrasings
# like "send an email to inimfon inviting him to the design meeting today"
# (no "about" at all) silently failed to match anything and fell through to
# the RAG chain. "that" is also a stop word (not just the verb-attached
# "letting her know that") for phrasing like "to John that I've completed
# my onboarding". Punctuation is a hard stop too -- a recipient name never
# legitimately spans a period, so "to Sandra. Subject line is ..." stops at
# "Sandra" instead of swallowing the whole rest of the sentence.
_EMAIL_RECIPIENT_PATTERN = re.compile(
    r"\bto\s+(.+?)(?=\s+\b(?:about|regarding|concerning|on|re:?|for|that|"
    r"inviting|telling|informing|asking|letting|updating|reminding)\b|[.!?]|$)",
    re.IGNORECASE,
)

# Once the recipient (if any) has been sliced off, whatever text remains is
# the topic -- but it usually still starts with a connector word/phrase
# ("about", "inviting him to", "letting her know that", ...) that reads
# awkwardly if left in a subject line or email body, so strip it off.
_EMAIL_LEADING_CONNECTORS = re.compile(
    r"^(?:about|regarding|concerning|on|re:?|for|that|"
    r"inviting\s+(?:him|her|them|us)?\s*(?:to)?|"
    r"telling\s+(?:him|her|them|us)?\s*(?:that|about)?|"
    r"informing\s+(?:him|her|them|us)?\s*(?:that|about)?|"
    r"asking\s+(?:him|her|them|us)?\s*(?:to|about)?|"
    r"letting\s+(?:him|her|them|us)?\s*know\s*(?:that|about)?|"
    r"updating\s+(?:him|her|them|us)?\s*(?:on|about)?|"
    r"reminding\s+(?:him|her|them|us)?\s*(?:to|about)?"
    r")\s+",
    re.IGNORECASE,
)

# Explicit dictation support: some people just state the subject and body
# outright ("Subject line is 'Project planning' and the body should ask
# 'when can we meet'") instead of a loose "about X" phrase. When present,
# these take priority over the generic topic extraction above, which would
# otherwise dump the whole dictated sentence into a mangled subject line.
_EMAIL_SUBJECT_QUOTED = re.compile(
    r'\bsubject(?:\s+line)?\s*(?:is|:)\s*"([^"]+)"', re.IGNORECASE
)
_EMAIL_SUBJECT_UNQUOTED = re.compile(
    r"\bsubject(?:\s+line)?\s*(?:is|:)\s*([^,.]+)", re.IGNORECASE
)
_EMAIL_BODY_INSTRUCTION = re.compile(
    r"\b(?:the\s+)?body\s+(?:should|is to|will|needs to)\s+(.+)", re.IGNORECASE
)


def _extract_explicit_subject_and_body(text: str) -> tuple[str, str]:
    """Pull an explicitly-dictated subject and/or body instruction out of
    text, e.g. 'Subject line is "Project planning" and the body should ask
    "when can we meet"' -> ('Project planning', 'ask "when can we meet"').
    Either can come back empty if not present."""
    subject_match = _EMAIL_SUBJECT_QUOTED.search(text) or _EMAIL_SUBJECT_UNQUOTED.search(text)
    subject = subject_match.group(1).strip(" ,.\"'") if subject_match else ""

    body_match = _EMAIL_BODY_INSTRUCTION.search(text)
    body_instruction = body_match.group(1).strip(" ,.\"'") if body_match else ""

    return subject, body_instruction


# Lets a user correct a wrongly-parsed recipient mid-conversation ("no, the
# recipient is Sandra") instead of only being able to cancel and start
# over. Deliberately checked before the generic continuation parsing in
# _continue_pending_email, and overwrites pending["recipient"] even if one
# was already set (bad or otherwise) -- a correction should always win.
_RECIPIENT_CORRECTION_PATTERN = re.compile(
    r"(?:no,?\s*)?(?:the\s+)?recipient\s+(?:is|should be|was supposed to be)\s+(?:just\s+)?(.+)",
    re.IGNORECASE,
)


def _parse_email_request(remainder: str) -> tuple[str, str]:
    """Pull (recipient, topic) out of the text following an email trigger
    verb, e.g. 'to inimfon inviting him to the design meeting today' ->
    ('inimfon', 'the design meeting today'). Either piece can come back
    empty -- an empty recipient falls back to a placeholder, an empty
    topic means route_task needs to ask for one before it can act."""
    remainder = remainder.strip(" ?.!")
    match = _EMAIL_RECIPIENT_PATTERN.search(remainder)
    if match:
        recipient = match.group(1).strip(" ,.")
        rest = remainder[match.end():].strip(" ,.")
    else:
        recipient = ""
        rest = remainder
    topic = _EMAIL_LEADING_CONNECTORS.sub("", rest, count=1).strip(" ,.")
    return recipient, topic


def _run_email_tool(
    mode: str, recipient: str, topic: str, llm=None, explicit_subject: str = ""
) -> str:
    """`explicit_subject` overrides the truncated-topic subject line when
    the user dictated one outright ("Subject line is ...") -- `topic` is
    still what gets passed as key_points to drive the body, since that's
    the actual content instruction even when a cleaner subject was given
    separately."""
    recipient_final = recipient or "[recipient]"
    subject = explicit_subject or (topic if len(topic) < 60 else topic[:57] + "...")
    tool = send_email if mode == "send" else draft_email
    result = tool(recipient=recipient_final, subject=subject, key_points=topic, llm=llm)
    if not recipient:
        result += (
            "\n\n_(No recipient was specified, so I used a placeholder -- "
            "let me know who this should go to and I'll redo it.)_"
        )
    return result


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

# Both triggers just require the verb immediately before "email" -- the
# recipient and topic are pulled out of whatever follows by
# _parse_email_request, which handles far more natural phrasing than a
# rigid "to X about Y" regex (e.g. "send an email to inimfon inviting him
# to the design meeting today", with no "about" at all).
#
# "Send" is a deliberately separate intent from "draft" -- draft_email only
# ever writes a local file, send_email actually attempts a real send (see
# email_integration.py for why that's routed to a fixed demo address
# rather than the parsed recipient). Deliberately NOT matching a bare
# "email X about Y" phrasing (no send/fire off/draft/etc. verb) -- too easy
# to false-positive on casual mentions of "email" as a noun (e.g. "check my
# email for updates about the project").
_EMAIL_DRAFT_TRIGGER = re.compile(
    r"\b(?:draft|write|prepare|compose)\b\s+(?:an |a )?email\b\s*(.*)", re.IGNORECASE
)
_EMAIL_SEND_TRIGGER = re.compile(
    r"\b(?:send|fire off)\b\s+(?:an |a )?email\b\s*(.*)", re.IGNORECASE
)

_CANCEL_PATTERN = re.compile(r"^\s*(cancel|never\s*mind|forget it|nvm)\s*[.!]?\s*$", re.IGNORECASE)


def _looks_like_new_intent(text: str) -> bool:
    """Cheap check for whether a message clearly starts a different,
    fully-formed request, used to decide whether to abandon a pending
    calendar/email task rather than treat this message as continuing it."""
    if any(p.search(text) for p in _TODO_PATTERNS):
        return True
    if any(p.search(text) for p in _REMEMBER_PATTERNS):
        return True
    if _EMAIL_DRAFT_TRIGGER.search(text) or _EMAIL_SEND_TRIGGER.search(text):
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


def _continue_pending_email(text: str, pending: dict, llm=None) -> tuple[str, dict | None]:
    if _CANCEL_PATTERN.match(text):
        return "No problem, I've dropped that email request.", None

    # A recipient correction ("no, the recipient is Sandra") always wins,
    # even overwriting an already-set (possibly wrong) recipient -- checked
    # first so it can't be shadowed by the generic parsing below, which
    # would otherwise just treat "the recipient is Sandra" as more topic
    # text and never actually fix anything.
    correction = _RECIPIENT_CORRECTION_PATTERN.search(text)
    if correction:
        pending["recipient"] = correction.group(1).strip(" ,.\"'")
        if pending.get("topic") or pending.get("explicit_subject"):
            return (
                _run_email_tool(
                    pending["mode"],
                    pending["recipient"],
                    pending.get("topic", ""),
                    llm=llm,
                    explicit_subject=pending.get("explicit_subject", ""),
                ),
                None,
            )
        return f"Got it -- what should the email to {pending['recipient']} be about?", pending

    # A continuation message is usually just the topic on its own ("inviting
    # him to the design meeting today"), but might still name a recipient we
    # don't have yet ("it's for inimfon, about the design meeting") -- try
    # the same recipient/topic parser before falling back to treating the
    # whole message as the topic. Also check for an explicitly dictated
    # subject/body, which takes priority over the generic topic text.
    recipient, topic = _parse_email_request(text)
    explicit_subject, body_instruction = _extract_explicit_subject_and_body(text)
    if body_instruction:
        topic = body_instruction
    elif not topic:
        topic = _EMAIL_LEADING_CONNECTORS.sub("", text.strip(" ,."), count=1).strip(" ,.")

    if recipient and not pending.get("recipient"):
        pending["recipient"] = recipient
    if topic:
        pending["topic"] = topic
    if explicit_subject:
        pending["explicit_subject"] = explicit_subject

    if pending.get("topic") or pending.get("explicit_subject"):
        return (
            _run_email_tool(
                pending["mode"],
                pending.get("recipient", ""),
                pending.get("topic", ""),
                llm=llm,
                explicit_subject=pending.get("explicit_subject", ""),
            ),
            None,
        )

    who_display = pending.get("recipient") or "them"
    return f"Got it. What should the email to {who_display} be about?", pending


def route_task(
    user_input: str, llm=None, pending_task: dict | None = None
) -> tuple[str | None, dict | None]:
    """Check user_input against known task intents.

    Returns (response, updated_pending_task):
    - response is a string if a tool ran or a clarifying question is being
      asked, or None if this message isn't a task at all (caller should
      fall through to the RAG chain).
    - updated_pending_task is a dict to store in session state and pass
      back in on the next call if a calendar/email request is still
      waiting on more info, or None if there's nothing pending anymore.

    `llm` is optional and used by draft_email/send_email to write a real
    body instead of a bare template; every other tool ignores it.
    """

    text = user_input.strip()

    # --- Continue an in-progress calendar/email request, if one exists and
    # this message doesn't clearly start something else entirely. ---
    if pending_task is not None:
        intent = pending_task.get("intent")
        if intent == "calendar" and not _looks_like_new_intent(text):
            return _continue_pending_calendar(text, pending_task)
        if intent == "email" and not _looks_like_new_intent(text):
            return _continue_pending_email(text, pending_task, llm=llm)
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
        # "on my/the/our calendar" is filler that shows up naturally in
        # phrasing like "add a meeting with Tim on my calendar to discuss
        # X" -- it's not part of who or topic, so drop it before extracting
        # either (otherwise it either gets swallowed into "who" or sits
        # between "who" and the topic connector and blocks the match).
        remainder = re.sub(
            r"\bon\s+(?:my|the|our)\s+calendar\b", "", remainder, flags=re.IGNORECASE
        ).strip(" ,.")
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

    # --- Email: send (real send attempt, falls back to draft-only) ---
    match = _EMAIL_SEND_TRIGGER.search(text)
    if match:
        remainder = match.group(1)
        recipient, topic = _parse_email_request(remainder)
        explicit_subject, body_instruction = _extract_explicit_subject_and_body(remainder)
        if body_instruction:
            topic = body_instruction
        if not topic and not explicit_subject:
            pending = {"intent": "email", "mode": "send", "recipient": recipient}
            who_display = recipient or "them"
            return f"Sure -- what should the email to {who_display} be about?", pending
        return _run_email_tool("send", recipient, topic, llm=llm, explicit_subject=explicit_subject), None

    # --- Email: draft only (never sent) ---
    match = _EMAIL_DRAFT_TRIGGER.search(text)
    if match:
        remainder = match.group(1)
        recipient, topic = _parse_email_request(remainder)
        explicit_subject, body_instruction = _extract_explicit_subject_and_body(remainder)
        if body_instruction:
            topic = body_instruction
        if not topic and not explicit_subject:
            pending = {"intent": "email", "mode": "draft", "recipient": recipient}
            who_display = recipient or "them"
            return f"Sure -- what should the email to {who_display} be about?", pending
        return _run_email_tool("draft", recipient, topic, llm=llm, explicit_subject=explicit_subject), None

    return None, None
