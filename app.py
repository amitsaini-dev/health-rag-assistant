import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Health RAG Assistant", page_icon="🏥")
st.title("🏥 Health RAG Assistant")
st.caption("Ask questions about Triage, EHR, and Primary Care")

# Load once and cache — so it doesn't reload on every message
@st.cache_resource
def load_rag():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )
    model = ChatOllama(model="llama3.2", temperature=0)
    return retriever, model

retriever, model = load_rag()

# Store chat history in streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_question := st.chat_input("Ask a health question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Step 1: Rewrite question if history exists
            if st.session_state.chat_history:
                rewrite_messages = [
                   SystemMessage(content="You are a question rewriter. Your ONLY job is to rewrite the user's follow-up question into a standalone search query under 10 words. Output the rewritten question only. No answers, no explanations.")
                ] + st.session_state.chat_history + [
                    HumanMessage(content=f"New Question: {user_question}")
                ]
                result = model.invoke(rewrite_messages)
                search_question = result.content.strip()
            else:
                search_question = user_question

            # Step 2: Retrieve docs
            relevant_docs = retriever.invoke(search_question)

            if not relevant_docs:
                answer = "I couldn't find relevant information in my documents for that question."
            else:
                # Step 3: Build prompt
                combined_input = f"""Answer the question using ONLY the documents below.
Question: {search_question}

Documents:
{chr(10).join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)])}

Answer directly and concisely.
"""
                # Step 4: Get answer
                answer_messages = [
                    SystemMessage(content="You are a helpful health assistant. Answer using ONLY the documents provided. Be concise — 2 to 3 sentences max. Never mention 'Document 1' or 'Document 2' in your answer."),
                    HumanMessage(content=combined_input),
                ]
                result = model.invoke(answer_messages)
                answer = result.content

            st.markdown(answer)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))