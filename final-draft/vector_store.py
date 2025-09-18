import json
import numpy as np
import faiss
import pickle
import os
import time
from sentence_transformers import SentenceTransformer

os.makedirs("vectordb", exist_ok=True)

docs = []
with open("rag_docs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            docs.append(json.loads(line))

print(f"Loaded {len(docs)} documents")

texts = [doc['text'] for doc in docs]
ids = [doc['id'] for doc in docs]

# Load the all-MiniLM-L6-v2 model
model = SentenceTransformer('all-MiniLM-L6-v2')

start_time = time.time()
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
elapsed = time.time() - start_time
print(f"Generated embeddings for {len(texts)} documents in {elapsed/60:.1f} min")

embeddings = np.array(embeddings).astype('float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vectordb/fitness_index.faiss")
with open("vectordb/fitness_docs.pkl", "wb") as f:
    pickle.dump({"ids": ids, "docs": docs}, f)

print("Vectorization complete and saved to vectordb/")