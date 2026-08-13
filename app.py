import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
import streamlit as st

from i18n import detect_language, translate_from_english, translate_to_english
from memory import get_long_term_facts, get_short_term_history
from personalization import get_profile, profile_to_prompt_snippet, save_profile
from router import route_task

st.set_page_config(page_title="Workplace Copilot", page_icon="🤖")
st.title("💼 Workplace Knowledge Assistant")

# All three features added on top of the original RAG + task-automation
# app (Personalization #5, Memory Management #6, Multilingual Support
# #15) deliberately favor deterministic, local processing over additional
# LLM calls -- same reasoning as router.py's keyword router: every extra
# model call is one more thing that can hit the free-tier Inference
# Providers quota that test_tool_calling.py already showed running out
# mid-test. Personalization is set via a plain sidebar form, long-term
# memory is only written on an explicit "remember that ..." command
# (router.py), and language detection/translation (i18n.py) run through
# local/free libraries rather than another model call.


@st.cache_resource
def load_retriever_and_llm():
    """Load the expensive, load-once-per-process resources: the embedding
    model, the FAISS index, and the LLM endpoint connection. Kept separate
    from the prompt/chain construction in build_rag_chain() below so that
    saving a personalization change (Feature #5) or picking up a new
    long-term fact (Feature #6) doesn't require reloading the FAISS index
    or reconnecting to the Hugging Face endpoint -- only the lightweight
    prompt object is rebuilt when that state changes."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.load_local(
        "faiss_workplace_index",
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )  # Retrieve top 3 chunks

    # Initialize Hugging Face LLM Endpoint
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature=0.1,  # Low temperature for precise responses
        provider="auto",  # let Hugging Face route to a provider that supports this model
    )
    # Wrap as a chat model: the router serves this model via the conversational
    # endpoint, not raw text-generation, so calls must go through ChatHuggingFace.
    llm = ChatHuggingFace(llm=llm)
    return retriever, llm


def build_rag_chain(retriever, llm, persona_snippet: str, long_term_snippet: str):
    """(Re)build the prompt and the RAG chain. This is cheap -- no I/O,
    just composing LangChain objects -- so it's safe to call on every
    turn. That's what lets the system prompt reflect the latest
    personalization (Feature #5) and long-term memory (Feature #6)
    without reloading the FAISS index or the LLM connection above.

    Short-term memory (also Feature #6) is handled separately: it's not
    baked into the system prompt text here, it's passed in per-turn as
    the chat_history variable via the MessagesPlaceholder below, since it
    changes on every single message rather than only when the user edits
    their profile or asks WorkBot to remember something.
    """
    system_prompt = (
        "You are WorkBot, a friendly enterprise workplace assistant. For questions "
        "about company policies, procedures, benefits, or anything that could be "
        "covered in the documentation below, answer using ONLY that context, and say "
        "clearly that it isn't covered in the documentation rather than guessing. For "
        "general conversation, greetings, or questions unrelated to company "
        "documentation, respond naturally and helpfully without needing the context.\n\n"
    )
    if persona_snippet:
        system_prompt += f"About this user: {persona_snippet}\n\n"
    if long_term_snippet:
        system_prompt += (
            f"Things this user has previously asked you to remember: "
            f"{long_term_snippet}\n\n"
        )
    system_prompt += "Context:\n{context}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


retriever, chat_llm = load_retriever_and_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_task" not in st.session_state:
    st.session_state.pending_task = None

# --- Personalization (Feature #5): sidebar form, saved locally, applied
# to every turn's system prompt below. ---
with st.sidebar:
    st.header("Your profile")
    st.caption("Saved locally and used to personalize WorkBot's answers.")
    profile = get_profile()
    with st.form("profile_form"):
        name = st.text_input("Name", value=profile["name"])
        department = st.text_input("Department", value=profile["department"])
        answer_style = st.radio(
            "Answer style",
            ["concise", "detailed"],
            index=0 if profile["answer_style"] == "concise" else 1,
        )
        if st.form_submit_button("Save"):
            save_profile(
                {"name": name, "department": department, "answer_style": answer_style}
            )
            st.success("Saved -- this applies starting with your next message.")
            profile = get_profile()

persona_snippet = profile_to_prompt_snippet(profile)
long_term_snippet = "; ".join(get_long_term_facts())
rag_chain = build_rag_chain(retriever, chat_llm, persona_snippet, long_term_snippet)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about company policies or processes..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # --- Multilingual support (Feature #15): detect the incoming
        # language and work in English internally (RAG index, router
        # patterns, and system prompt are all English-only), then
        # translate the final answer back before it's shown. ---
        detected_lang = detect_language(user_input)
        english_input = translate_to_english(user_input, detected_lang)

        # --- Memory Management (Feature #6), short-term tier: replay the
        # last few turns (excluding the one just appended above, which is
        # passed separately as {input}) into the model's context. ---
        chat_history = get_short_term_history(st.session_state.messages[:-1])

        # Check for a task-automation intent (to-do, calendar event, email
        # draft, or "remember that ...") before falling through to the RAG
        # chain. This is a plain function call, not a model call, so it
        # doesn't touch the free Inference Providers quota. pending_task
        # carries an in-progress calendar request across turns (e.g. still
        # waiting on a date/time) so follow-up messages get merged into it
        # instead of being treated as brand-new, context-free requests.
        task_result, st.session_state.pending_task = route_task(
            english_input, llm=chat_llm, pending_task=st.session_state.pending_task
        )
        if task_result is not None:
            answer = task_result
        else:
            response = rag_chain.invoke(
                {"input": english_input, "chat_history": chat_history}
            )
            answer = response["answer"]

        answer = translate_from_english(answer, detected_lang)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
