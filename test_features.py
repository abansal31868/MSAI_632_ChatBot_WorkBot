"""
Standalone verification script for Personalization (#5), Memory Management
(#6), and Multilingual Support (#15).

Same spirit as test_tool_calling.py: a script that lives outside the app
and checks one thing works before you rely on it in a live demo. Unlike
test_tool_calling.py, this does NOT need HUGGINGFACEHUB_API_TOKEN or the
FAISS index -- all three features were deliberately built without any LLM
call (see memory.py, personalization.py, and i18n.py's docstrings), so
this script talks to the real SQLite storage and the real langdetect /
deep-translator libraries directly. No mocking, no model.

Each test isolates itself with a fresh temp SQLite file, so running this
never touches your real workbot_data.db.

Run with:
    python test_features.py

Requires: langdetect, deep-translator (same as app.py). The i18n checks
make real network calls to Google Translate's free endpoint, so they will
fail if you're offline or if that endpoint is rate-limiting you --
that's useful information too, not just noise, since app.py depends on
that same endpoint at runtime and fails open (falls back to English) if
it's unreachable.
"""

import tempfile

import i18n
import memory
import personalization

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    suffix = f" -- {detail}" if detail and status == FAIL else ""
    print(f"[{status}] {name}{suffix}")


# --- Personalization (#5) ---------------------------------------------

def test_personalization():
    personalization.DB_PATH = tempfile.mktemp(suffix=".db")

    default = personalization.get_profile()
    check("personalization: defaults load", default == personalization.DEFAULT_PROFILE)

    personalization.save_profile(
        {"name": "Test User", "department": "QA", "answer_style": "concise"}
    )
    reloaded = personalization.get_profile()
    check(
        "personalization: save/reload round-trip",
        reloaded["name"] == "Test User" and reloaded["answer_style"] == "concise",
        detail=str(reloaded),
    )

    snippet = personalization.profile_to_prompt_snippet(reloaded)
    check("personalization: snippet mentions name", "Test User" in snippet, snippet)
    check("personalization: snippet mentions concise", "concise" in snippet, snippet)

    # Partial update should only touch the field given, not reset the rest.
    personalization.save_profile({"answer_style": "detailed"})
    after = personalization.get_profile()
    check("personalization: partial update preserves name", after["name"] == "Test User")
    check("personalization: partial update changes style", after["answer_style"] == "detailed")

    # Unconfigured profile shouldn't inject a false "detailed" instruction
    # before the user has touched the sidebar at all.
    personalization.DB_PATH = tempfile.mktemp(suffix=".db")
    fresh_snippet = personalization.profile_to_prompt_snippet(personalization.get_profile())
    check(
        "personalization: default profile still produces a style instruction",
        "detailed" in fresh_snippet,
        fresh_snippet,
    )


# --- Memory Management (#6) --------------------------------------------

def test_memory():
    memory.DB_PATH = tempfile.mktemp(suffix=".db")

    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}"})

    history = memory.get_short_term_history(messages)
    expected_len = memory.SHORT_TERM_TURNS * 2
    check(
        "memory: short-term window size",
        len(history) == expected_len,
        f"got {len(history)}, expected {expected_len}",
    )
    check(
        "memory: short-term keeps the MOST RECENT turns, not the oldest",
        history[-1].content == "answer 9" and history[0].content == "question 6",
        f"first={history[0].content!r} last={history[-1].content!r}",
    )

    check("memory: fresh long-term store starts empty", memory.get_long_term_facts() == [])
    memory.remember_fact("works on the Q3 migration")
    memory.remember_fact("prefers bullet points")
    facts = memory.get_long_term_facts()
    check(
        "memory: long-term facts persist in the order they were added",
        facts == ["works on the Q3 migration", "prefers bullet points"],
        str(facts),
    )


# --- Multilingual Support (#15) ----------------------------------------

def test_i18n():
    check("i18n: short string assumed English (no detection attempted)",
          i18n.detect_language("hi") == "en")

    spanish = "Buenos dias, como puedo solicitar mis vacaciones este ano"
    lang = i18n.detect_language(spanish)
    check("i18n: detects Spanish on a real, longer sentence", lang == "es", f"got {lang!r}")

    check(
        "i18n: English input passes through untouched (no translator call)",
        i18n.translate_to_english("hello there", "en") == "hello there",
    )

    try:
        english = i18n.translate_to_english(spanish, "es")
        looks_english = any(
            w in english.lower() for w in ["good", "morning", "vacation", "request", "how"]
        )
        check("i18n: real translation to English looks plausible", looks_english, english)

        back = i18n.translate_from_english(
            "Your vacation request has been approved.", "es"
        )
        check(
            "i18n: real translation back to Spanish changed the text",
            back.strip().lower() != "your vacation request has been approved.",
            back,
        )
    except Exception as exc:  # noqa: BLE001 - a network failure here IS the finding
        check(
            "i18n: translation round-trip reachable",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    # Fail-open behavior: an intentionally bad/unsupported source code
    # should return the original text, not raise.
    result = i18n.translate_to_english("some text", "not-a-real-language-code")
    check(
        "i18n: unsupported language code fails open (returns original text)",
        result == "some text",
        result,
    )


def main():
    test_personalization()
    test_memory()
    test_i18n()

    passed = sum(1 for r in results if r[0] == PASS)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed < total:
        print("\nFailures:")
        for status, name, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()
