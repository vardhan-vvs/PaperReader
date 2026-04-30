# vector_store.py — Chunking, FAISS index management, and disk persistence

import hashlib
import json
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    CHARS_PER_PAGE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAISS_INDEX_PATH,
    METADATA_FILE,
)


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_name: str = "") -> tuple[list[str], list[dict]]:
    """
    Split *text* into overlapping chunks using sentence-aware splitting.

    Uses RecursiveCharacterTextSplitter which tries paragraph breaks → sentences
    → words → characters in order, keeping semantic units intact.

    Returns:
        (chunks, metadatas) — metadatas[i] holds {"source": filename, "page": N}
        for citation purposes.
    """
    if not text.strip():
        return [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    char_offsets = _compute_offsets(text, chunks)

    metadatas = [
        {
            "source": source_name,
            "page": max(1, offset // CHARS_PER_PAGE + 1),
        }
        for offset in char_offsets
    ]
    return chunks, metadatas


def _compute_offsets(text: str, chunks: list[str]) -> list[int]:
    """Estimate the character offset of each chunk inside the original text."""
    offsets = []
    cursor = 0
    for chunk in chunks:
        idx = text.find(chunk[:60], cursor)
        if idx == -1:
            idx = cursor
        offsets.append(idx)
        cursor = max(cursor, idx)
    return offsets


# ── Build / merge ──────────────────────────────────────────────────────────────

def build_vector_store(
    text: str, embedder: Embeddings, source_name: str = ""
) -> FAISS | None:
    """
    Chunk *text* and build a FAISS store with source + page metadata per chunk.

    The metadata survives retrieval, so rag.py can cite which paper (and
    approximate page) each piece of context came from.
    """
    chunks, metadatas = chunk_text(text, source_name)
    if not chunks:
        return None
    return FAISS.from_texts(chunks, embedder, metadatas=metadatas)


def merge_vector_stores(stores: list[FAISS]) -> FAISS | None:
    """
    Merge multiple FAISS stores into one for 'All papers' queries.

    Source metadata is preserved, so cross-paper answers can still be cited
    back to the correct paper.
    """
    if not stores:
        return None
    base = stores[0]
    for vs in stores[1:]:
        base.merge_from(vs)
    return base


# ── Disk persistence ───────────────────────────────────────────────────────────

def save_index(name: str, vs: FAISS) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vs.save_local(os.path.join(FAISS_INDEX_PATH, name))


def load_index(name: str, embedder: Embeddings) -> FAISS | None:
    index_path = os.path.join(FAISS_INDEX_PATH, name)
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embedder, allow_dangerous_deserialization=True
        )
    return None


def delete_index(name: str) -> bool:
    """Remove a FAISS index folder from disk. Returns True if it existed."""
    index_path = os.path.join(FAISS_INDEX_PATH, name)
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        return True
    return False


# ── Metadata ───────────────────────────────────────────────────────────────────

def load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_metadata(meta: dict) -> None:
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ── File hashing ───────────────────────────────────────────────────────────────

def file_hash(file_bytes: bytes) -> str:
    """SHA-256 of raw bytes — content-based cache key for re-upload detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def index_name_for_hash(fhash: str) -> str:
    return f"index_{fhash[:16]}"