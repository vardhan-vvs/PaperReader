# rag.py — RAG logic: retrieval, citation, streaming, memory, comparison mode

from collections import defaultdict
from typing import Generator

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    COMPARISON_KEYWORDS,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_GPU,
    LLM_TIMEOUT,
    RETRIEVAL_K,
)

# ── Sentinel responses ─────────────────────────────────────────────────────────
_NO_DATA = "No text data available. Please upload at least one PDF first."
_NOT_FOUND = "The document does not contain information relevant to this question."


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm(streaming: bool = False) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=0,
        num_ctx=LLM_NUM_CTX,
        num_gpu=LLM_NUM_GPU,
        request_timeout=LLM_TIMEOUT,
        streaming=streaming,
    )


# ── Prompt builders ────────────────────────────────────────────────────────────

def _standard_prompt(context: str, question: str) -> str:
    return (
        "You are a research paper assistant. Answer the question using only "
        "the context below. If the answer is not in the context, say: "
        f'"{_NOT_FOUND}"\n\n'
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )


def _comparison_prompt(grouped_context: dict[str, str], question: str) -> str:
    """Build a prompt that structures context paper-by-paper for comparative queries."""
    sections = "\n\n".join(
        f"=== {paper} ===\n{ctx}" for paper, ctx in grouped_context.items()
    )
    return (
        "You are a research paper assistant comparing multiple papers. "
        "Answer the question below by addressing each paper individually, "
        "then summarise the key similarities and differences. "
        "Use only the context provided. "
        f'If a paper lacks relevant info, say "Not found in [paper name]."\n\n'
        f"Context by paper:\n{sections}\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )


def _build_history_messages(
    history: list[dict], system_prompt: str
) -> list:
    """Convert the session chat history into LangChain message objects."""
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ── Citation formatter ─────────────────────────────────────────────────────────

def _format_citations(docs) -> str:
    """
    Build a compact citation block from retrieved document metadata.

    Groups chunks by (source, page) and returns a markdown-style string like:
        Sources:
        • paper_a.pdf — p. 3, 5
        • paper_b.pdf — p. 7
    """
    seen: dict[str, set] = defaultdict(set)
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        seen[source].add(page)

    if not seen:
        return ""

    lines = ["**Sources:**"]
    for source, pages in seen.items():
        sorted_pages = ", ".join(str(p) for p in sorted(pages))
        lines.append(f"- {source} — p. {sorted_pages}")
    return "\n".join(lines)


# ── Comparison intent detection ────────────────────────────────────────────────

def is_comparison_query(query: str) -> bool:
    """Return True if the query appears to ask for a cross-paper comparison."""
    q = query.lower()
    return any(kw in q for kw in COMPARISON_KEYWORDS)


# ── Public API ─────────────────────────────────────────────────────────────────

def rag_answer(
    query: str,
    vs: FAISS | None,
    history: list[dict] | None = None,
    paper_stores: dict[str, FAISS] | None = None,
) -> tuple[str, str]:
    """
    Answer *query* using RAG and return (answer, citations).

    Args:
        query:        The user's question.
        vs:           The FAISS store to retrieve from (per-paper or global).
        history:      Prior chat messages for conversation memory.
        paper_stores: Per-paper stores used only in comparison mode.

    Returns:
        (answer_text, citation_block) — both are plain strings.
        citation_block is empty if no sources were found.
    """
    if vs is None:
        return _NO_DATA, ""

    history = history or []
    use_comparison = (
        is_comparison_query(query)
        and paper_stores
        and len(paper_stores) > 1
    )

    if use_comparison:
        return _comparison_answer(query, paper_stores, history)
    else:
        return _standard_answer(query, vs, history)


def _standard_answer(
    query: str, vs: FAISS, history: list[dict]
) -> tuple[str, str]:
    docs = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K}).invoke(query)
    if not docs:
        return _NOT_FOUND, ""

    context = "\n\n".join(d.page_content for d in docs)
    system = _standard_prompt(context, "")  # context baked into system
    messages = _build_history_messages(history, system)
    messages.append(HumanMessage(content=query))

    llm = _get_llm()
    answer = llm.invoke(messages)
    text = answer.content if hasattr(answer, "content") else str(answer)
    return text, _format_citations(docs)


def _comparison_answer(
    query: str, paper_stores: dict[str, FAISS], history: list[dict]
) -> tuple[str, str]:
    """Retrieve from each paper separately, then answer with a structured prompt."""
    grouped_context: dict[str, str] = {}
    all_docs = []

    for paper_name, paper_vs in paper_stores.items():
        docs = paper_vs.as_retriever(search_kwargs={"k": 2}).invoke(query)
        if docs:
            grouped_context[paper_name] = "\n\n".join(d.page_content for d in docs)
            all_docs.extend(docs)

    if not grouped_context:
        return _NOT_FOUND, ""

    prompt_text = _comparison_prompt(grouped_context, query)
    system = (
        "You are a research paper assistant comparing multiple papers. "
        "Always address each paper individually before summarising."
    )
    messages = _build_history_messages(history, system)
    messages.append(HumanMessage(content=prompt_text))

    llm = _get_llm()
    answer = llm.invoke(messages)
    text = answer.content if hasattr(answer, "content") else str(answer)
    return text, _format_citations(all_docs)


def rag_stream(
    query: str,
    vs: FAISS | None,
    history: list[dict] | None = None,
    paper_stores: dict[str, FAISS] | None = None,
) -> Generator[str, None, None]:
    """
    Streaming variant of rag_answer.

    Yields answer tokens one by one for use with st.write_stream().
    Citations are yielded as a final block after the answer completes.

    Usage in Streamlit:
        citations_holder = st.empty()
        answer = st.write_stream(rag_stream(query, vs, history, paper_stores))
    """
    if vs is None:
        yield _NO_DATA
        return

    history = history or []
    use_comparison = (
        is_comparison_query(query)
        and paper_stores
        and len(paper_stores) > 1
    )

    if use_comparison:
        grouped_context: dict[str, str] = {}
        all_docs = []
        for paper_name, paper_vs in paper_stores.items():
            docs = paper_vs.as_retriever(search_kwargs={"k": 2}).invoke(query)
            if docs:
                grouped_context[paper_name] = "\n\n".join(d.page_content for d in docs)
                all_docs.extend(docs)

        if not grouped_context:
            yield _NOT_FOUND
            return

        prompt_text = _comparison_prompt(grouped_context, query)
        system = (
            "You are a research paper assistant comparing multiple papers. "
            "Always address each paper individually before summarising."
        )
        messages = _build_history_messages(history, system)
        messages.append(HumanMessage(content=prompt_text))
        citations = _format_citations(all_docs)
    else:
        docs = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K}).invoke(query)
        if not docs:
            yield _NOT_FOUND
            return
        context = "\n\n".join(d.page_content for d in docs)
        system = _standard_prompt(context, "")
        messages = _build_history_messages(history, system)
        messages.append(HumanMessage(content=query))
        citations = _format_citations(docs)

    llm = _get_llm(streaming=True)
    for chunk in llm.stream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield token

    # Yield citations as a trailing block separated from the answer
    if citations:
        yield f"\n\n---\n{citations}"