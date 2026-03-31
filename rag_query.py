# this file will load the FAISS and chunks, retrieve the relevant chunks
# it will construct a prompt and ask Ollama to answer the question and return the answer

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_ollama import OllamaLLM

index = faiss.read_index("fv_index.faiss")
texts = pickle.load(open("chunks.pkl", "rb"))

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
llm = OllamaLLM(model = "llama3:instruct")

def pr_rag(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    data, indices = index.search(query_embedding, k=5)

    relevant_chunks = [texts[i] for i in indices[0]]
    context = "\n".join(relevant_chunks)
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

if __name__ == "__main__":
    query = "What is this paper about?"
    answer = pr_rag(query)
    print("Answer:", answer)