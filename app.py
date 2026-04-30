# app.py — Streamlit entry point. UI and session orchestration only.
# Run with: streamlit run app.py

import streamlit as st
from paper_handler import extract_text

from embeddings import get_embedder
from rag import is_comparison_query, rag_stream
from vector_store import (
    build_vector_store,
    delete_index,
    file_hash,
    index_name_for_hash,
    load_index,
    load_metadata,
    merge_vector_stores,
    save_index,
    save_metadata,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Research Paper Reader", layout="wide")

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "paper_vs": {},         # filename → FAISS store
    "paper_hashes": {},     # filename → SHA-256
    "global_vs": None,      # merged store for "All papers"
    "chat_history": {},     # filename (or "All papers") → [{"role","content"}, ...]
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Singletons ─────────────────────────────────────────────────────────────────
embedder = get_embedder()
disk_meta = load_metadata()     # {sha256: index_folder_name}


# ── Sidebar — paper manager ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Paper Reader")
    st.caption("Chat with your research papers")
    st.divider()

    # File uploader lives in the sidebar for a cleaner main area
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.caption("Upload one or more PDF files")
    st.divider()

    # Loaded papers list with per-paper remove buttons
    if st.session_state.paper_vs:
        st.markdown("**Loaded papers**")
        for fname in list(st.session_state.paper_vs.keys()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}", help=fname)
            if col2.button("✕", key=f"remove_{fname}", help=f"Remove {fname}"):
                # Remove from session
                fhash = st.session_state.paper_hashes.pop(fname, None)
                st.session_state.paper_vs.pop(fname, None)
                st.session_state.chat_history.pop(fname, None)

                # Remove from disk and metadata
                if fhash and fhash in disk_meta:
                    delete_index(disk_meta.pop(fhash))
                    save_metadata(disk_meta)

                # Rebuild global store without the removed paper
                st.session_state.global_vs = merge_vector_stores(
                    list(st.session_state.paper_vs.values())
                )
                st.rerun()
    else:
        st.info("No papers loaded yet.")


# ── Process uploaded files ─────────────────────────────────────────────────────
if uploaded_files:
    newly_indexed = False

    for file in uploaded_files:
        raw_bytes = file.read()
        fhash = file_hash(raw_bytes)

        if st.session_state.paper_hashes.get(file.name) == fhash:
            continue  # Already loaded with same content

        if fhash in disk_meta:
            with st.spinner(f"Loading saved index for **{file.name}**…"):
                vs = load_index(disk_meta[fhash], embedder)
            if vs is not None:
                st.session_state.paper_vs[file.name] = vs
                st.session_state.paper_hashes[file.name] = fhash
                st.toast(f"✅ Loaded {file.name} from disk")
                newly_indexed = True
                continue

        with st.spinner(f"Reading **{file.name}**…"):
            raw_text = extract_text(file)
            text = raw_text if isinstance(raw_text, str) else "\n".join(raw_text)

        if not text.strip():
            st.warning(f"No text found in **{file.name}**. Skipping.")
            continue

        with st.spinner(f"Building embeddings for **{file.name}**…"):
            vs = build_vector_store(text, embedder, source_name=file.name)

        if vs is None:
            st.warning(f"Could not build index for **{file.name}**. Skipping.")
            continue

        iname = index_name_for_hash(fhash)
        save_index(iname, vs)
        disk_meta[fhash] = iname
        save_metadata(disk_meta)

        st.session_state.paper_vs[file.name] = vs
        st.session_state.paper_hashes[file.name] = fhash
        st.toast(f"✅ Indexed {file.name}")
        newly_indexed = True

    if newly_indexed and st.session_state.paper_vs:
        with st.spinner("Rebuilding global index…"):
            st.session_state.global_vs = merge_vector_stores(
                list(st.session_state.paper_vs.values())
            )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Research Paper Reader")

if not st.session_state.paper_vs:
    st.info("👈 Upload PDFs from the sidebar to get started.")
    st.stop()

# Paper selector
options = ["All papers"] + list(st.session_state.paper_vs.keys())
selected_paper = st.selectbox("Query scope", options)

# Show a badge if the query looks like a comparison and "All papers" is selected
query_hint = st.empty()

# Ensure a history list exists for this scope
if selected_paper not in st.session_state.chat_history:
    st.session_state.chat_history[selected_paper] = []

history = st.session_state.chat_history[selected_paper]

# Clear chat button (inline with selector)
col_sel, col_clr = st.columns([5, 1])
with col_clr:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history[selected_paper] = []
        st.rerun()

# Render conversation history
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the paper(s)…")

if query:
    # Show comparison badge when applicable
    if selected_paper == "All papers" and is_comparison_query(query):
        query_hint.info(
            "🔀 Comparison mode — answering paper-by-paper then summarising."
        )

    # Resolve which FAISS store to use
    vs = (
        st.session_state.global_vs
        if selected_paper == "All papers"
        else st.session_state.paper_vs.get(selected_paper)
    )

    with st.chat_message("user"):
        st.markdown(query)
    history.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        # Stream the response token-by-token
        streamed = st.write_stream(
            rag_stream(
                query=query,
                vs=vs,
                history=history[:-1],           # exclude the query we just added
                paper_stores=st.session_state.paper_vs if selected_paper == "All papers" else None,
            )
        )

    # st.write_stream returns the full concatenated string
    history.append({"role": "assistant", "content": streamed})