import os

FAISS_INDEX_PATH = "faiss_indexes"
METADATA_FILE = os.path.join(FAISS_INDEX_PATH, "metadata.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHARS_PER_PAGE = 3000   # used to estimate page numbers from char offsets

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama3.2:3b"
LLM_BASE_URL = "http://localhost:11434"
LLM_NUM_CTX = 4096     # increased to accommodate history + context
LLM_NUM_GPU = 28
LLM_TIMEOUT = 300

RETRIEVAL_K = 4 # chunks to retrieve

COMPARISON_KEYWORDS = [
    "compare", "comparison", "contrast", "difference", "similar",
    "common", "across", "each paper", "all papers", "both papers",
    "versus", "vs", "between",
]