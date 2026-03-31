import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

index = faiss.read_index("fv_index.faiss")
texts = pickle.load(open("chunks.pkl", "rb"))

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
query = "What is the title of this paper?"
query_embedding = model.encode([query], convert_to_numpy=True)

data, indices = index.search(query_embedding, k=5)

print("\n Top 5 relevant chunks:") 
for i in indices[0]:
    print(f"Chunk {i}: {texts[i]}")
    print("------------")
    