from pyexpat import model

import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
import pdfplumber

print(torch.cuda.is_available())      # True = GPU detected
print(torch.cuda.get_device_name(0))  # Shows GPU name
print(torch.cuda.device_count())      # Number of GPUs
class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Using device: {self.model.device}")

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]
        # return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.encode(text).tolist()
        # return self.model.embed_query(text)

def extract_text_from_pdf(pdf_path):
    chunks = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks += text
    return chunks

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, 
    #     chunk_overlap=100,
    #     separators=["\n\n", "\n", ".", " ", ""]
    # )

    # docs = splitter.split_documents(pages)
    # return docs

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
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "The document does not contain this info."
    
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = ChatOllama(
        model="llama3:instruct",
        base_url="http://localhost:11434", 
        temperature=0,
        num_ctx=4096,
        request_timeout=300
        )
    
    prompt = f"""
    You are a research paper assistant. You will answer the question only using the context below.
    If you don't know the answer, say you don't know. Do not make up an answer.
    If the answer is not in the context, say "The document does not contain this info."
    context:
    {context}
    question:
    {query}
    Answer:
    """

    answer = llm.invoke(prompt)
    return answer

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
                text = extract_text_from_pdf(file)
            
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