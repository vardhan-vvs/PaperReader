# 📄 Paper Reader

A local, privacy-first RAG (Retrieval-Augmented Generation) application that lets you chat with your research papers. Upload one or more PDFs and ask questions — per paper, or across all of them at once. Everything runs on your machine with no API costs and no data leaving your system.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Ollama](https://img.shields.io/badge/Ollama-local-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

- **Multi-paper upload** — load as many PDFs as you need in one session
- **Per-paper or cross-paper querying** — switch scope with a single dropdown
- **Comparison mode** — ask things like *"what are the common approaches across these papers?"* and get a structured, paper-by-paper breakdown automatically
- **Persistent embeddings** — indexes are saved to disk and reloaded on next launch; no re-embedding the same file twice
- **Source citations** — every answer includes the paper name and estimated page number of the retrieved context
- **Streaming responses** — answers appear token-by-token instead of after a long wait
- **Conversation memory** — follow-up questions like *"can you elaborate on point 2?"* work naturally within a session
- **Paper manager sidebar** — view loaded papers and remove any with one click
- **GPU acceleration** — uses CUDA automatically if available, falls back to CPU silently

---

## 🏗️ Architecture

```
paper_reader/
├── app.py            # Streamlit UI — session orchestration only
├── rag.py            # Retrieval, prompting, streaming, comparison mode
├── vector_store.py   # Chunking, FAISS build/merge, disk persistence
├── embeddings.py     # MiniLM wrapper + cached singleton
├── config.py         # All tuneable constants in one place
├── paper_handler.py  # PDF text extraction (bring your own)
└── requirements.txt  # Python dependencies
```

Each module has a single responsibility. To swap the LLM, edit `rag.py`. To change the embedding model, edit `embeddings.py`. To tune chunking or retrieval, edit `config.py` — nothing else needs to change.

---

## 🔧 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Required for `X \| None` type syntax |
| [Ollama](https://ollama.com) | Latest | Must be running locally |
| NVIDIA GPU | Optional | CUDA 12.x for GPU acceleration |

### Install and start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull llama3.2:3b

# Start the server (runs on http://localhost:11434)
ollama serve
```

> You can use any Ollama-compatible model. Update `LLM_MODEL` in `config.py` to switch — for example `llama3.1:8b` for better quality or `phi3:mini` for faster responses on CPU.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/paper-reader.git
cd paper-reader
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch + FAISS (hardware-dependent)

**GPU (NVIDIA CUDA 12.x):**
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install faiss-gpu==1.9.0
```

**CPU only:**
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu==1.9.0
```

> `torch` and `faiss` are excluded from `requirements.txt` because they ship different binaries for GPU vs CPU. The app auto-detects your hardware at runtime — no code changes needed.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

### Uploading papers

Use the **sidebar** to upload one or more PDFs. Each paper is:
1. Extracted to text via `paper_handler.py`
2. Split into overlapping chunks with `RecursiveCharacterTextSplitter`
3. Embedded with `all-MiniLM-L6-v2` and indexed in FAISS
4. Saved to disk under `faiss_indexes/` so it never needs re-indexing

On subsequent launches, previously indexed papers load instantly from disk.

### Querying

Select a scope from the dropdown:

| Scope | Behaviour |
|---|---|
| **All papers** | Retrieves from the merged global index |
| **Specific paper** | Retrieves only from that paper's index |

Each scope maintains its own **independent chat history**, so switching papers doesn't mix up your conversations.

### Comparison mode

When querying **All papers**, certain keywords automatically trigger comparison mode:

> *compare, contrast, difference, similar, common, across, each paper, both papers, versus, vs, between*

In comparison mode, the app retrieves from each paper's index independently and instructs the LLM to address each paper individually before summarising. A `🔀 Comparison mode` badge appears when it activates.

**Example queries:**
- *"What are the common limitations mentioned across these papers?"*
- *"Compare the evaluation methods used in each paper."*
- *"What is the difference in datasets used between these papers?"*

### Managing papers

Each paper listed in the sidebar has a **✕** button. Clicking it:
- Removes the paper from the current session
- Deletes its FAISS index folder from disk
- Updates the metadata file
- Rebuilds the global index without it

---

## ⚙️ Configuration

All tuneable values live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `CHARS_PER_PAGE` | `3000` | Characters per page (for citation page estimation) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model name |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model name |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_NUM_CTX` | `4096` | LLM context window size |
| `LLM_NUM_GPU` | `28` | GPU layers offloaded to VRAM |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `COMPARISON_KEYWORDS` | *(list)* | Keywords that trigger comparison mode |

---

## 📁 Data Storage

Indexes are stored locally in the `faiss_indexes/` directory:

```
faiss_indexes/
├── metadata.json              # Maps SHA-256 file hashes → index folder names
├── index_a3f1c8b2d4e67890/    # FAISS index for paper A
│   ├── index.faiss
│   └── index.pkl
└── index_b9e2d1a5f7c34512/    # FAISS index for paper B
    ├── index.faiss
    └── index.pkl
```

Indexes are keyed by the **SHA-256 hash of the file's raw bytes**, not the filename. Re-uploading a modified version of the same file automatically triggers a fresh index build, while re-uploading an unchanged file loads the existing index instantly.

> Add `faiss_indexes/` to your `.gitignore` — indexes are large binary files and are machine-generated.

---

## 🛠️ Customisation

**Swap the LLM model:**
```python
# config.py
LLM_MODEL = "llama3.1:8b"    # higher quality
LLM_MODEL = "phi3:mini"       # faster on CPU
LLM_MODEL = "mistral:7b"      # good all-rounder
```

**Swap the embedding model:**
```python
# config.py
EMBEDDING_MODEL = "all-mpnet-base-v2"    # higher quality, slower
EMBEDDING_MODEL = "all-MiniLM-L12-v2"   # balanced
```

**Point to a remote Ollama instance:**
```python
# config.py
LLM_BASE_URL = "http://192.168.1.100:11434"
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.45.1 | Web UI |
| `langchain` | 1.2.16 | RAG orchestration |
| `langchain-core` | 1.3.2 | Message types, base classes |
| `langchain-community` | 0.3.31 | FAISS vector store integration |
| `langchain-ollama` | 1.1.0 | Ollama LLM integration |
| `sentence-transformers` | 5.4.1 | Local embedding model |
| `pymupdf` | 1.27.2.3 | PDF text extraction |
| `torch` | 2.6.0 | ML backend for embeddings |
| `faiss-gpu` / `faiss-cpu` | 1.9.0 | Vector similarity search |

---

## 🗺️ Roadmap

- [ ] Cross-encoder re-ranking for improved retrieval precision
- [ ] Adjustable retrieval `k` via UI slider
- [ ] Query history panel per paper
- [ ] Support for `.docx`, `.txt`, and `.md` file formats
- [ ] Export chat history to PDF or markdown

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.