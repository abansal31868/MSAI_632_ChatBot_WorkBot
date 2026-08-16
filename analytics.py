import re
import sqlite3
from datetime import datetime
import uuid

DB_PATH = "workbot_analytics.db"


def anonymize_text(text: str) -> str:
    """Remove PII-like content before writing telemetry."""
    if not text:
        return ""
    redacted = text
    redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", redacted)
    redacted = re.sub(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b", "[PHONE]", redacted)
    return redacted[:2000]


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interaction_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            request_latency_ms REAL,
            retrieval_docs_count INTEGER,
            answer_length_chars INTEGER,
            token_estimate INTEGER,
            tool_used INTEGER,
            tool_success INTEGER,
            completion_rate REAL,
            feedback TEXT,
            user_message TEXT,
            response_text TEXT
        )
        """
    )
    return conn


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def record_interaction(
    *,
    session_id: str | None,
    user_message: str,
    response_text: str,
    request_latency_ms: float,
    retrieval_docs_count: int = 0,
    tool_used: bool = False,
    tool_success: bool = False,
    completion_rate: float = 1.0,
    feedback: str | None = None,
):
    """Persist anonimized interaction telemetry for monitoring and tuning."""
    if not session_id:
        session_id = uuid.uuid4().hex[:12]

    conn = _get_db()
    conn.execute(
        """
        INSERT INTO interaction_metrics (
            timestamp,
            session_id,
            request_latency_ms,
            retrieval_docs_count,
            answer_length_chars,
            token_estimate,
            tool_used,
            tool_success,
            completion_rate,
            feedback,
            user_message,
            response_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            session_id,
            round(float(request_latency_ms), 2),
            int(retrieval_docs_count),
            len(response_text or ""),
            estimate_tokens(response_text or ""),
            1 if tool_used else 0,
            1 if tool_success else 0,
            float(completion_rate),
            feedback,
            anonymize_text(user_message),
            anonymize_text(response_text),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_metrics(limit: int = 5):
    conn = _get_db()
    rows = conn.execute(
        """
        SELECT timestamp, request_latency_ms, retrieval_docs_count,
               token_estimate, tool_success, completion_rate, feedback
        FROM interaction_metrics
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def get_aggregate_metrics():
    conn = _get_db()
    row = conn.execute(
        """
        SELECT
            AVG(request_latency_ms) AS avg_latency,
            AVG(completion_rate) AS avg_completion_rate,
            AVG(token_estimate) AS avg_tokens,
            SUM(CASE WHEN tool_success = 1 THEN 1 ELSE 0 END) AS successful_tools,
            COUNT(*) AS total_interactions
        FROM interaction_metrics
        """
    ).fetchone()
    conn.close()
    return row
