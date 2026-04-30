# embeddings.py — SentenceTransformer wrapper for LangChain + Streamlit caching

import streamlit as st
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class MiniLMEmbeddings(Embeddings):
    """
    Thin LangChain-compatible wrapper around a SentenceTransformers model.
    Device is chosen automatically: CUDA GPU if available, else CPU.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name, device=_detect_device())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@st.cache_resource
def get_embedder() -> MiniLMEmbeddings:
    """Singleton — loaded once per Streamlit server process."""
    return MiniLMEmbeddings()