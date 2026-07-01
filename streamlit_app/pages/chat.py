"""
Chat page for the Streamlit application.
"""

import streamlit as st

from streamlit_app.utils.api_client import query_backend, document_upload_rag

# Configure page settings
st.set_page_config(
    page_title="LangGraph Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None
    }
)

# Initialize logout confirmation state
if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

# Header with logout button
col1, col2 = st.columns([10, 2])
with col2:
    st.write("")  # Spacer
    if st.button("🔒 Logout", use_container_width=True):
        st.session_state.show_logout_confirm = True

# Logout confirmation dialog
if st.session_state.show_logout_confirm:
    st.warning("Are you sure you want to logout?")
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("✅ Yes, logout"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Redirect to home page
            st.switch_page("Home.py")
    with col_cancel:
        if st.button("❌ Cancel"):
            st.session_state.show_logout_confirm = False

st.title("💬 LangGraph Chat")

# Document upload section
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    file_description = None
    if uploaded_file:
        file_description = st.text_input(
            "📄 Describe your document (required)",
            max_chars=300,
            placeholder="E.g. LangGraph tutorial with workflows and code examples"
        )

        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}

        file_key = f"{uploaded_file.name}_{file_description}"

        if file_description:
            if file_key not in st.session_state.uploaded_files:
                # Upload file if not already uploaded
                success = document_upload_rag(uploaded_file, file_description)
                if success:
                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.session_state.uploaded_files[file_key] = True
                else:
                    st.error(f"Document Upload Failed: {uploaded_file.name}")
            else:
                st.info(f"Uploaded: {uploaded_file.name}")
        else:
            st.warning("Please describe your document before uploading.")

# Check authentication
if "session_id" not in st.session_state:
    st.warning("Please login first.")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask a question...")

# Process user input and get response
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = query_backend(user_input, st.session_state["jwt_token"])
    # response is a dict: {content, confidence, sources}
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()  # Rerun script to display updated messages

# Display chat history
for role, text in st.session_state.chat_history:
    if role == "assistant":
        # text is expected to be a dict
        if isinstance(text, dict):
            content = text.get("content")
            confidence = text.get("confidence")
            sources = text.get("sources") or []
            retrieval_analytics = text.get("retrieval_analytics") or []

            st.chat_message(role).write(content)
            if confidence is not None:
                with st.chat_message(role):
                    st.markdown(f"**Confidence:** {confidence}%")
            if sources:
                with st.chat_message(role):
                    st.markdown("**Sources:**")
                    for s in sources:
                        st.markdown(f"- {s}")
            # Retrieval Analytics panel in the sidebar (hidden by default)
            with st.sidebar.expander("Retrieval Analytics", expanded=False):
                st.markdown("**System Info**")
                try:
                    from src.config.settings import RERANKER_MODEL, RETRIEVAL_TOP_K, FINAL_TOP_K, USE_RERANKER
                    from src.llms.ollama import MODEL_NAME as LLM_MODEL
                    from src.llms.embeddings import embeddings as EMB_PROVIDER
                    emb_name = getattr(EMB_PROVIDER, "__class__", type(EMB_PROVIDER)).__name__
                except Exception:
                    RERANKER_MODEL = None
                    RETRIEVAL_TOP_K = None
                    FINAL_TOP_K = None
                    USE_RERANKER = None
                    LLM_MODEL = None
                    emb_name = None

                st.markdown(f"- Hybrid Search: **Enabled**")
                st.markdown(f"- Embedding Model: **{emb_name or 'unknown'}**")
                st.markdown(f"- LLM Model: **{LLM_MODEL or 'unknown'}**")
                st.markdown(f"- Reranker Model: **{RERANKER_MODEL or 'none'}**")
                st.markdown(f"- Top-K Retrieved: **{RETRIEVAL_TOP_K or 'n/a'}**")
                st.markdown(f"- Final Documents Passed to LLM: **{FINAL_TOP_K or 'n/a'}**")

                st.markdown("---")
                st.markdown("**Retrieved Chunks (post-rerank)**")
                if not retrieval_analytics:
                    st.info("No retrieval analytics available for this response.")
                else:
                    rows = []
                    for rank, cand in enumerate(retrieval_analytics, start=1):
                        meta = cand.get("meta", {}) or {}
                        filename = meta.get("filename") or meta.get("source") or "unknown"
                        page = meta.get("page")
                        chunk_id = meta.get("chunk_id")
                        method = meta.get("retrieval_method")
                        retrieval_score = meta.get("retrieval_score")
                        bm25_score = retrieval_score if (method and method.upper().startswith("BM")) else None
                        semantic_score = retrieval_score if (method and not method.upper().startswith("BM")) else None
                        rrf_score = meta.get("rrf_score")
                        reranker_score = meta.get("reranker_score")
                        rows.append({
                            "final_ranking": rank,
                            "filename": filename,
                            "page": page if page is not None else "-",
                            "chunk_id": chunk_id if chunk_id is not None else "-",
                            "retrieval_method": method or "-",
                            "semantic_similarity": semantic_score if semantic_score is not None else "-",
                            "bm25_score": bm25_score if bm25_score is not None else "-",
                            "rrf_score": rrf_score if rrf_score is not None else "-",
                            "crossencoder_score": reranker_score if reranker_score is not None else "-",
                        })

                    import pandas as pd

                    df = pd.DataFrame(rows)
                    st.dataframe(df)
        else:
            st.chat_message(role).write(text)
    else:
        st.chat_message(role).write(text)
