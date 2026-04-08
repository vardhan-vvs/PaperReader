from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from paper_handler import docs

# load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

texts = [doc.page_content for doc in docs]

embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print("Number of vectors in index:", index.ntotal)
faiss.write_index(index, "fv_index.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Index and chunks saved successfully.")
