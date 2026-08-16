import re
from typing import Sequence

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"developer prompt",
    r"override safety",
    r"jailbreak",
    r"act as if",
    r"you are now",
    r"ignore all prior rules",
    r"disregard the policy",
    r"bypass the guardrails",
    r"pretend you are",
]

HARMFUL_PATTERNS = [
    r"bomb",
    r"explosive",
    r"weapon",
    r"malware",
    r"phishing",
    r"steal credentials",
    r"bypass security",
    r"self-harm",
    r"suicide",
    r"violent attack",
]


def detect_prompt_injection(text: str) -> bool:
    """Flag attempts to override the model's instructions or system behavior."""
    normalized = (text or "").lower()
    return any(re.search(pattern, normalized) for pattern in PROMPT_INJECTION_PATTERNS)


def detect_harmful_content(text: str) -> bool:
    """Flag requests that ask for dangerous or unsafe actions."""
    normalized = (text or "").lower()
    return any(re.search(pattern, normalized) for pattern in HARMFUL_PATTERNS)


def has_relevant_context(question: str, documents: Sequence) -> bool:
    """Return True when retrieved documents appear to match the user's question."""
    if not documents:
        return False

    question_terms = set(
        re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (question or "").lower())
    )
    if not question_terms:
        return True

    doc_text = " ".join((doc.page_content if hasattr(doc, "page_content") else str(doc)).lower() for doc in documents)
    matches = sum(1 for term in question_terms if term and term in doc_text)
    return matches > 0


def validate_response(answer: str, question: str, documents: Sequence) -> str:
    """Reduce hallucinations by refusing to answer when the evidence is weak."""
    if not answer or not answer.strip():
        return "I couldn’t generate a reliable answer from the available documentation. Please ask again with more specific details."

    answer_lower = answer.lower()
    safe_markers = [
        "not covered in the documentation",
        "not enough information",
        "i don't know",
        "i’m not sure",
        "i can't confirm",
        "unable to verify",
    ]
    if any(marker in answer_lower for marker in safe_markers):
        return answer

    if not has_relevant_context(question, documents):
        return "I couldn't find enough relevant documentation to answer that reliably. Please provide more detail or ask about a policy/process that is in the documents."

    return answer


def evaluate_user_request(text: str) -> str | None:
    """Return a safe refusal message if the request violates guardrails."""
    if detect_prompt_injection(text):
        return "I can help with workplace questions using company documentation, but I can’t assist with requests that try to override instructions, bypass safeguards, or manipulate the system."
    if detect_harmful_content(text):
        return "I can’t assist with requests related to harmful, dangerous, or policy-violating activities."
    return None
