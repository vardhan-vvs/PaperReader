from sentence_transformers import SentenceTransformer
import torch

print("CUDA:", torch.cuda.is_available())

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

sentences = ['This is an example sentence', 'Each sentence is converted']
embeddings = model.encode(sentences)
print("Vector size:", len(embeddings[0]))