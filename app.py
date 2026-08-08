import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
import streamlit as st

st.set_page_config(page_title="Workplace Copilot", page_icon="🤖")
st.title("💼 Workplace Knowledge Assistant")


@st.cache_resource
def load_rag_chain():
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

    system_prompt = (
        "You are WorkBot, a friendly enterprise workplace assistant. For questions "
        "about company policies, procedures, benefits, or anything that could be "
        "covered in the documentation below, answer using ONLY that context, and say "
        "clearly that it isn't covered in the documentation rather than guessing. For "
        "general conversation, greetings, or questions unrelated to company "
        "documentation, respond naturally and helpfully without needing the context.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


rag_chain = load_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about company policies or processes..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": user_input})
        st.markdown(response["answer"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )
