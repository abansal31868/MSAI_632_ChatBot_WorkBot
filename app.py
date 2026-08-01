import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_token_here"

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
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        temperature=0.1,  # Low temperature for precise responses
    )

    system_prompt = (
        "You are an enterprise workplace assistant. Answer the question using ONLY "
        "the context below. If unknown, state that it's not found in documentation.\n\n"
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