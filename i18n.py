"""
Multilingual support for WorkBot (Feature #15: Multilingual Support).

Approach: detect the incoming message's language with a lightweight,
fully local library (no model download, no network call), and if it
isn't English, translate it to English before it reaches the RAG chain
and the deterministic router in router.py, then translate WorkBot's
answer back into the user's language before it's shown. This keeps the
FAISS index, the router's regex patterns, and the system prompt entirely
in English -- only the two translation edges are language-aware --
rather than needing a multilingual embedding model, a multilingual
re-index of the same documents, and multilingual variants of every
pattern in router.py.

Language detection: langdetect (pure Python, runs locally, no API key).

Translation: deep-translator's GoogleTranslator backend, which calls
Google Translate's free web endpoint -- deliberately NOT the paid Google
Cloud Translation API, no billing account or API key required. This is a
best-effort, unofficial free tier, same spirit as the rest of this
project's "free resources" constraint: it can rate-limit or change
without notice, so every call here is wrapped and falls back to the
original text rather than crashing the chat on a translation hiccup.

Known limitation, worth stating plainly: short-term memory (memory.py)
replays each turn's *displayed* text -- i.e. the user's original-language
message and WorkBot's already-translated-back answer -- into the model's
context on later turns, rather than the internal English versions. This
avoids a third translation call per turn, and Llama-3.1-8B-Instruct has
reasonable multilingual comprehension on its own, but it does mean memory
quality for non-English conversations depends on the base model's
multilingual ability rather than on this module.
"""

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# langdetect's ISO 639-1 code for English.
ENGLISH = "en"

# Below this length, language detection on short strings (e.g. "hi", "ok",
# "thanks") is unreliable -- langdetect needs some real text to work with.
# Short messages are assumed to be English rather than risking a
# wrong-language false positive that garbles a two-word reply.
MIN_CHARS_FOR_DETECTION = 12


def detect_language(text: str) -> str:
    """Best-effort language detection. Returns an ISO 639-1 code, or "en"
    if detection wasn't attempted (short text) or failed outright."""
    if len(text.strip()) < MIN_CHARS_FOR_DETECTION:
        return ENGLISH
    try:
        return detect(text)
    except LangDetectException:
        return ENGLISH


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text into English. Falls back to the original text on
    any failure (network issue, unsupported language code, free-tier rate
    limit) so a translation problem degrades to 'treat it as English'
    rather than crashing the chat."""
    if source_lang == ENGLISH:
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception:
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate WorkBot's English answer back into the user's language.
    Same fail-open behavior as translate_to_english: on any error, return
    the English text rather than showing an error to the user."""
    if target_lang == ENGLISH:
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text
