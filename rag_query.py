# this file will load the FAISS and chunks, retrieve the relevant chunks
# it will construct a prompt and ask Ollama to answer the question and return the answer

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_ollama import OllamaLLM

