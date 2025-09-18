import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import ssl

# Fix SSL issues if needed
ssl._create_default_https_context = ssl._create_unverified_context

class RAGRetriever:
    def __init__(self, index_path="vectordb/fitness_index.faiss", docs_path="vectordb/fitness_docs.pkl"):
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the documents and mapping
        with open(docs_path, "rb") as f:
            data = pickle.load(f)
            self.docs = data["docs"]
            self.ids = data["ids"]
        
        # Load the model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_context(self, query, top_k=3):
        """
        Retrieve relevant context for a query.
        Returns formatted context ready to be used in a prompt.
        """
        results = self.retrieve(query, top_k)
        
        # Format context from retrieved documents
        context_parts = []
        
        for i, result in enumerate(results):
            doc = result["document"]
            score = result["score"]
            source_type = doc.get("meta", {}).get("source_type", "general")
            
            context_part = f"[Document {i+1} - {source_type}]\n{doc['text']}"
            context_parts.append(context_part)
            
        context = "\n\n".join(context_parts)
        return context
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve relevant documents for a query.
        Returns the raw results with documents and scores.
        """
        # Encode the query
        query_vector = self.model.encode([query])[0].astype('float32')
        query_vector = np.array([query_vector])
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Get the documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.docs):  # Safety check
                doc = self.docs[idx]
                results.append({
                    "document": doc,
                    "score": float(distances[0][i])
                })
        
        return results

# For testing independently
if __name__ == "__main__":
    retriever = RAGRetriever()
    query = "What are good sources of protein for vegetarians?"
    
    # Test the context formatting
    context = retriever.get_context(query)
    print(f"CONTEXT FOR: '{query}'\n")
    print(context)
    
    print("\n\nRAW RESULTS:")
    results = retriever.retrieve(query)
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        print(f"Type: {result['document'].get('meta', {}).get('source_type', 'unknown')}")
        print(f"Text: {result['document']['text'][:200]}...")