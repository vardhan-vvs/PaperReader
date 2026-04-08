import streamlit as st
# import torch
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from paper_handler import extract_text
import os

# print(torch.cuda.is_available())      # True = GPU detected
# print(torch.cuda.get_device_name(0))  # Shows GPU name
# print(torch.cuda.device_count())      # Number of GPUs

faiss_index_path = "faiss_indexes"

def save_faiss_index(name, vs):
    os.makedirs(faiss_index_path, exist_ok=True)
    vs.save_local(os.path.join(faiss_index_path, name))
    
def load_faiss_index(name, embedder):
    index_path = os.path.join(faiss_index_path, name)
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    return None
class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

def build_vector_store(texts):
    chunks = []
    chunk_size = 800
    chunk_overlap = 100
    for i in range(0, len(texts), chunk_size - chunk_overlap):
        chunk = texts[i:i + chunk_size]
        chunks.append(chunk)
    
    if not chunks:
        return None
    
    embedder = MiniLMEmbeddings()
    vs = FAISS.from_texts(chunks, embedder)
    
    return vs

def rag_answer(query, vs):
    if vs is None:
        return "No text data available to answer the question."
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    if not docs:
        return "The document does not contain this info."
    
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434", 
        temperature=0,
        num_ctx=2048,
        num_gpu=28,
        request_timeout=300
        )
    
    prompt = f"""
    You are a research paper assistant. You will answer the question only using the context below.
    If the answer is not in the context, say "The document does not contain this info."
    context:
    {context}
    question:
    {query}
    Answer:
    """

    answer = llm.invoke(prompt)
    return answer.content if hasattr(answer, "content") else str(answer)

st.set_page_config(page_title="Research Paper reader", layout="wide")
st.title("Paper Reader - Chat with your research papers")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "paper_vs" not in st.session_state:
    st.session_state.paper_vs = {}
if "global_vs" not in st.session_state:
    st.session_state.global_vs = None


uploaded_file = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_file is not None and len(uploaded_file) > 0:
    global_text = ""
    
    for file in uploaded_file:
        if file.name not in st.session_state.paper_vs:
            with st.spinner(f"Reading {file.name}..."):
                text = extract_text(file)
            
            with st.spinner(f"Building embeddings for {file.name}..."):
                vs = build_vector_store(text)
            if vs is None:
                st.warning(f"No text found in {file.name}. Skipping this file.")
                continue
                
            st.session_state.paper_vs[file.name] = vs
            global_text += text + "\n"
            
    if global_text.strip() == "":
        st.warning("No text found in any of the uploaded papers. Global vector store will not be created.")
    else:
        with st.spinner("updating global embeddings..."):
            st.session_state.global_vs = build_vector_store(global_text)

    st.success("All papers processed successfully! You can now ask questions about any paper.")

options = ["All papers"] + list(st.session_state.paper_vs.keys())
selected_paper = st.selectbox("Select a paper", options)


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
query = st.chat_input("Ask a question about the paper")
if query is not None:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    if selected_paper == "All papers":
        vs = st.session_state.global_vs
    else:
        vs = st.session_state.paper_vs.get(selected_paper)
    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            answer = rag_answer(query, vs)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})