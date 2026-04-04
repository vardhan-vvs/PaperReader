import streamlit as st
# import faiss
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import os
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

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

def buidl_vector_store(texts):
    chunks = []
    chunk_size = 1000
    chunk_overlap = 200
    for i in range(0, len(texts), chunk_size - chunk_overlap):
        chunk = texts[i:i + chunk_size]
        chunks.append(chunk)
    
    embedder = MiniLMEmbeddings()
    vs = FAISS.from_texts(chunks, embedder)
    
    return vs

def rag_answer(query, vs):
    retriever = vs.as_retriever(search_type = "similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = OllamaLLM(model="llama3:instruct")
    
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
if "vs" not in st.session_state:
    st.session_state.vs = None


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Reading the paper..."):
        text = extract_text_from_pdf(uploaded_file)
    
    with st.spinner("Building embeddings ..."):
        st.session_state.vs = buidl_vector_store(text)

    st.success("Paper processed successfully! You can now ask questions about the paper.")


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
query = st.chat_input("Ask a question about the paper")
if query and st.session_state.vs is not None:
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            answer = rag_answer(query, st.session_state.vs)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})