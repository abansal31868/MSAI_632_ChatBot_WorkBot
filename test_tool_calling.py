"""
Standalone reliability check for LLM-driven tool calling through
Hugging Face's Inference Providers routing.

This is NOT part of the app. It exists to answer one question before we
commit to an architecture for API/Tool Integration + Task Automation:
does `ChatHuggingFace(llm=...).bind_tools([...])` reliably pick the right
tool (or no tool) given our current model + provider="auto" routing?

Run with:
    python test_tool_calling.py

Requires HUGGINGFACEHUB_API_TOKEN to be set in your shell, same as app.py.
"""

import os
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

MODEL_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
TRIALS_PER_PROMPT = 3  # run each prompt a few times since provider routing can vary


# --- Dummy tools -----------------------------------------------------------
# Names/descriptions mirror the shape of the real tools we'd build for
# Task Automation (add_todo, create_calendar_event) plus one that should
# clearly NOT be triggered by an unrelated question, to check false positives.

@tool
def add_todo(task: str, due_date: str) -> str:
    """Add a to-do item for the user. Use this when the user asks to remember,
    track, or add a task/reminder/to-do, optionally with a due date."""
    return f"[TOOL CALLED] add_todo(task={task!r}, due_date={due_date!r})"


@tool
def create_calendar_event(title: str, when: str) -> str:
    """Create a calendar event. Use this when the user asks to schedule a
    meeting, call, or event at a specific time."""
    return f"[TOOL CALLED] create_calendar_event(title={title!r}, when={when!r})"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this only when the user
    explicitly asks about weather."""
    return f"[TOOL CALLED] get_weather(city={city!r})"


TOOLS = [add_todo, create_calendar_event, get_weather]

# --- Test prompts ------------------------------------------------------------
# (prompt, expected_tool_name_or_None)
TEST_CASES = [
    ("Add a to-do to send the Q3 report by Friday.", "add_todo"),
    ("Remind me to call the vendor tomorrow.", "add_todo"),
    ("Schedule a meeting with Sarah tomorrow at 2pm.", "create_calendar_event"),
    ("Set up a call with the design team for next Monday at 10am.", "create_calendar_event"),
    ("What's the weather like in Paris today?", "get_weather"),
    ("What is the capital of France?", None),  # should NOT call a tool
    ("How are you?", None),  # should NOT call a tool
]


def main():
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        raise SystemExit(
            "HUGGINGFACEHUB_API_TOKEN is not set in this shell. "
            "Run: export HUGGINGFACEHUB_API_TOKEN=\"your_hf_token_here\""
        )

    llm = HuggingFaceEndpoint(
        repo_id=MODEL_REPO_ID,
        task="text-generation",
        temperature=0.1,
        provider="auto",
    )
    chat = ChatHuggingFace(llm=llm)
    chat_with_tools = chat.bind_tools(TOOLS)

    total = 0
    correct = 0
    results = []

    for prompt, expected in TEST_CASES:
        for trial in range(1, TRIALS_PER_PROMPT + 1):
            total += 1
            try:
                response = chat_with_tools.invoke(prompt)
                tool_calls = getattr(response, "tool_calls", None) or []
                got = tool_calls[0]["name"] if tool_calls else None
                ok = got == expected
                correct += int(ok)
                results.append((prompt, trial, expected, got, ok, None))
            except Exception as exc:  # noqa: BLE001 - we want to see any failure mode
                results.append((prompt, trial, expected, "ERROR", False, str(exc)[:200]))

    print("\n=== Tool-calling reliability results ===\n")
    for prompt, trial, expected, got, ok, err in results:
        status = "OK " if ok else "FAIL"
        line = f"[{status}] trial {trial} | expected={expected!r:22} got={got!r:22} | {prompt}"
        print(line)
        if err:
            print(f"        error: {err}")

    print(f"\n{correct}/{total} correct ({100 * correct / total:.0f}%)\n")
    print(
        "Rule of thumb: below ~80-85% here, the LLM-driven agent approach is too "
        "flaky for a live demo and we should lean on the deterministic keyword "
        "router instead (or use it as a fallback when tool_calls comes back empty "
        "or wrong)."
    )


if __name__ == "__main__":
    main()
