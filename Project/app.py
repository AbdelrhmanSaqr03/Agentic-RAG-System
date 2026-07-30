"""
Streamlit entry point for the Agentic RAG Knowledge Assistant.
Provides the chat interface, knowledge base management sidebar, and
document upload workflow.
"""

import os
import time

import streamlit as st

from config import settings
from rag.loaders import DocumentLoader
from rag.chunking import DocumentChunker
from rag.vector_database import VectorDatabase
from rag.retriever import Retriever
from agent.agent import AgenticRAG
from agent.memory import ConversationMemory
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title=settings.app_title,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize all required Streamlit session state variables."""
    if "vector_database" not in st.session_state:
        st.session_state.vector_database = VectorDatabase()
        st.session_state.vector_database.load()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "agent" not in st.session_state:
        retriever = Retriever(st.session_state.vector_database)
        st.session_state.agent = AgenticRAG(retriever, st.session_state.memory)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def rebuild_knowledge_base() -> None:
    """Reload documents, re-chunk, re-embed, and rebuild the FAISS index."""
    with st.spinner("Rebuilding knowledge base..."):
        loader = DocumentLoader()
        documents = loader.load_all()

        if not documents:
            st.warning("No supported documents found in the documents folder.")
            return

        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        vector_database = VectorDatabase()
        vector_database.build_from_documents(chunks)

        st.session_state.vector_database = vector_database
        retriever = Retriever(vector_database)
        st.session_state.agent = AgenticRAG(retriever, st.session_state.memory)

    st.success(f"Knowledge base rebuilt from {len(documents)} document(s), {len(chunks)} chunks.")


def handle_file_upload(uploaded_files) -> None:
    """Save uploaded files into the documents directory."""
    if not uploaded_files:
        return

    saved_count = 0
    for uploaded_file in uploaded_files:
        destination = os.path.join(settings.paths.documents_dir, uploaded_file.name)
        with open(destination, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())
        saved_count += 1

    st.success(f"Saved {saved_count} file(s) to the documents folder.")


def render_sidebar() -> None:
    """Render the sidebar with knowledge base controls and file upload."""
    with st.sidebar:
        st.title("📚 " + settings.app_title)
        st.markdown("An agentic assistant that plans, retrieves, and cites sources.")

        st.divider()
        st.subheader("Knowledge Base")

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, CSV, XLSX)",
            type=["pdf", "docx", "txt", "csv", "xlsx"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            handle_file_upload(uploaded_files)

        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            rebuild_knowledge_base()

        st.divider()
        st.subheader("Conversation")

        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()

        st.divider()
        ready = st.session_state.vector_database.store is not None
        status_label = "✅ Ready" if ready else "⚠️ Empty (build the knowledge base)"
        st.caption(f"Knowledge base status: {status_label}")


def render_chat_history() -> None:
    """Render all previous chat turns stored in session state."""
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("sources_text"):
                st.caption(f"📎 Sources: {turn['sources_text']}")
            if turn.get("execution_time") is not None:
                st.caption(f"⏱️ Answered in {turn['execution_time']}s")


def handle_user_question(question: str) -> None:
    """Run the agent on a new user question and render the response."""
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.perf_counter()
            try:
                result = st.session_state.agent.ask(question)
            except Exception as exc:
                logger.error("Agent failed to answer: %s", exc)
                result = {
                    "answer": "An error occurred while generating the answer. "
                              "Please check your API key and try again.",
                    "sources_text": "",
                }
            elapsed = round(time.perf_counter() - start_time, 3)

        st.markdown(result["answer"])
        if result.get("sources_text"):
            st.caption(f"📎 Sources: {result['sources_text']}")
        st.caption(f"⏱️ Answered in {elapsed}s")

    st.session_state.chat_history.append({
        "question": question,
        "answer": result["answer"],
        "sources_text": result.get("sources_text", ""),
        "execution_time": elapsed,
    })


def main() -> None:
    """Application entry point."""
    initialize_session_state()
    render_sidebar()

    st.header("💬 Chat with your Knowledge Base")
    render_chat_history()

    question = st.chat_input("Ask a question about your documents...")
    if question:
        handle_user_question(question)


if __name__ == "__main__":
    main()
