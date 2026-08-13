"""
User personalization for WorkBot (Feature #5: Personalization).

Rather than trying to infer preferences from free-text chat -- fragile,
and easy to mis-parse ("I like concise answers" vs. "keep it short" vs.
"no fluff" are all the same request phrased differently) -- personalization
is set explicitly through a small Streamlit sidebar form and stored
locally in the same SQLite file tools.py and memory.py already use
(workbot_data.db). This keeps personalization fully deterministic and
free of any extra model calls, consistent with the router-first
philosophy documented in router.py and test_tool_calling.py: let the LLM
focus on answering questions, not on classifying the user's intent when a
plain form field can do it reliably instead.

The stored profile is turned into a short instruction block and folded
into the system prompt on every turn (see app.py's build_rag_chain), so
the underlying LLM adapts its tone, level of detail, and department
framing to the user without needing a separate personalization model or
service.
"""

import sqlite3

DB_PATH = "workbot_data.db"  # same local file tools.py and memory.py use

DEFAULT_PROFILE = {
    "name": "",
    "department": "",
    "answer_style": "detailed",  # "concise" or "detailed"
}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return conn


def get_profile() -> dict:
    """Load the stored profile, filling in defaults for anything not yet
    set (e.g. first run, empty database)."""
    conn = _get_db()
    rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
    conn.close()
    profile = dict(DEFAULT_PROFILE)
    profile.update({k: v for k, v in rows})
    return profile


def save_profile(profile: dict) -> None:
    """Upsert each field of the profile. Called from app.py's sidebar
    form on submit."""
    conn = _get_db()
    for key, value in profile.items():
        conn.execute(
            "INSERT INTO user_profile (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    conn.commit()
    conn.close()


def profile_to_prompt_snippet(profile: dict) -> str:
    """Turn the stored profile into a short instruction block for the
    system prompt. Returns an empty string if nothing meaningful is set,
    so an unconfigured profile doesn't add noise (or a false "detailed
    answers" instruction) to every prompt before the user has touched the
    sidebar."""
    lines = []
    if profile.get("name"):
        lines.append(
            f"The user's name is {profile['name']}; address them by name "
            "when it's natural to do so."
        )
    if profile.get("department"):
        lines.append(
            f"The user works in {profile['department']}; when a policy "
            "question is ambiguous across departments, prioritize "
            "documentation relevant to that department."
        )
    if profile.get("answer_style") == "concise":
        lines.append(
            "The user prefers concise answers: 2-4 sentences, no "
            "unnecessary preamble."
        )
    elif profile.get("answer_style") == "detailed":
        lines.append(
            "The user prefers detailed, thorough answers with relevant "
            "context and examples."
        )
    return " ".join(lines)
