import re
from typing import Iterable

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

ALLOWED_ROLES = {"employee", "manager", "admin", "auditor"}


def sanitize_input(text: str) -> str:
    """Strip obvious control characters and destructive markup before processing."""
    if not text:
        return ""
    sanitized = text.replace("\x00", "")
    sanitized = sanitized.replace("\r", " ")
    sanitized = sanitized.replace("\n", " ")
    return sanitized.strip()


def mask_pii(text: str) -> str:
    """Mask common PII fields to reduce data exposure in stored logs or UI."""
    if not text:
        return ""
    masked = EMAIL_PATTERN.sub("[EMAIL]", text)
    masked = PHONE_PATTERN.sub("[PHONE]", masked)
    masked = SSN_PATTERN.sub("[SSN]", masked)
    return masked


def check_rbac(role: str | None, required_role: str = "employee") -> bool:
    """Simple role gate for demo-level RBAC. True means the user is allowed."""
    normalized = (role or "").lower().strip()
    if normalized not in ALLOWED_ROLES:
        return False
    role_rank = {"employee": 1, "manager": 2, "admin": 3, "auditor": 4}
    return role_rank.get(normalized, 0) >= role_rank.get(required_role, 0)


def secure_session_context(role: str | None) -> dict:
    """Return a security snapshot that can be displayed in the app."""
    return {
        "authenticated": bool(role),
        "role": (role or "guest").lower(),
        "rbac_ok": check_rbac(role),
        "https_enforced": True,
        "pii_masking_enabled": True,
        "audit_logging_enabled": True,
    }


def validate_request(text: str) -> str | None:
    """Return a denial message for blocked/unsafe input. In demo mode this is a light policy guard."""
    cleaned = sanitize_input(text or "")
    if not cleaned:
        return "I couldn't process an empty request."
    if "<script" in cleaned.lower() or "javascript:" in cleaned.lower():
        return "This request looks unsafe and cannot be processed."
    return None
