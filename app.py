import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
st.title("RAG based Research Paper Assistant")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
llm = OllamaLLM(model = "llama3:instruct")

def process_pdf(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)
    return chunks

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, 
    #     chunk_overlap=100,
    #     separators=["\n\n", "\n", ".", " ", ""]
    # )

    # docs = splitter.split_documents(pages)
    # return docs

def build_faiss(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def rag_answer(query, index, chunks):
    query_embedding = model.encode([query], convert_to_numpy=True)
    data, indices = index.search(query_embedding, k=5)

    relevant_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
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

st.sidebar.header("Upload Research Paper")
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file:
    st.success("PDF uploaded successfully!")
    
    chunks = process_pdf(pdf_file)
    index, chunks = build_faiss(chunks)
    
    st.sidebar.write(f"Total chunks created: {len(chunks)}")
    
    st.subheader("Ask a question about the paper")
    user_query = st.text_input("Enter your question here:")
    
    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Retrieving answer..."):
                answer = rag_answer(user_query, index, chunks)
            st.markdown("### Answer:")
            st.write("Answer:", answer)
else:
    st.warning("Please upload a PDF file to get started.")