"""
Conversation memory for WorkBot (Feature #6: Memory Management).

Two tiers, matching the proposal:

- Short-term memory: the last few turns of the current session, replayed
  into the LLM's context window on every call so WorkBot can resolve
  pronouns and follow-up questions ("what about last year's version?")
  within a single conversation. This lives entirely in Streamlit's
  st.session_state -- nothing to persist, it's gone when the session ends.

- Long-term memory: a small set of durable facts about the user that
  should survive across sessions (e.g. "prefers short answers", "works on
  the Q3 migration project"), stored locally in the same SQLite file
  tools.py already creates (workbot_data.db), so there's a single local
  data store for the whole app rather than several scattered files.
  Facts are added deterministically via an explicit "remember that ..."
  command (see router.py's _REMEMBER_PATTERNS) rather than inferred by
  the LLM from ordinary conversation -- same reasoning as router.py's
  choice of a keyword router over LLM tool-calling: a misfire here means
  WorkBot silently "remembers" something the user never asked it to,
  which is a worse failure mode than occasionally requiring the user to
  phrase the request explicitly.

Design tradeoff, deliberate: retrieval in app.py still runs on the raw
(English-translated) user input, not a history-rewritten query. A
"condense the question using chat history" step is the standard way to
make retrieval itself history-aware, but it costs one extra LLM call per
turn against the same free-tier quota that test_tool_calling.py already
showed running out mid-test. Instead, chat_history is passed straight
into the answer-writing prompt (via a MessagesPlaceholder) so the model
can still resolve a follow-up using the previous turns' text, even though
the retriever itself only sees the current message. This covers the
common case (a follow-up that shares vocabulary with the prior question)
without a second network call, at the cost of not helping with follow-ups
that are worded completely differently from the original question.
"""

import sqlite3
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage

DB_PATH = "workbot_data.db"  # same local file tools.py uses

# How many prior user/assistant turns to replay into the LLM's context.
# Kept small on purpose: every extra turn is extra tokens against a
# free-tier model, and stale context can also mislead the model on a
# genuinely unrelated follow-up rather than help it.
SHORT_TERM_TURNS = 4


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def get_short_term_history(messages: list[dict]) -> list:
    """Convert the Streamlit-style message list
    ([{"role": "user"|"assistant", "content": ...}, ...]) into a list of
    LangChain message objects for the last SHORT_TERM_TURNS exchanges,
    suitable for a ChatPromptTemplate's MessagesPlaceholder("chat_history").

    Expects `messages` to NOT include the current, in-progress user turn --
    app.py appends that separately as the chain's {input}.
    """
    trimmed = messages[-(SHORT_TERM_TURNS * 2):]
    history = []
    for m in trimmed:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history


def remember_fact(fact: str) -> str:
    """Persist a fact to long-term memory. Called by router.py when the
    user explicitly asks WorkBot to remember something."""
    conn = _get_db()
    conn.execute(
        "INSERT INTO long_term_memory (fact, created_at) VALUES (?, ?)",
        (fact.strip(), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return f"Got it, I'll remember: {fact.strip()}"


def get_long_term_facts(limit: int = 20) -> list[str]:
    """Return the most recent long-term facts, oldest first, for injecting
    into the system prompt. Capped at `limit` so a long-running deployment
    doesn't eventually dump hundreds of facts into every single prompt."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT fact FROM long_term_memory ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [r[0] for r in reversed(rows)]
