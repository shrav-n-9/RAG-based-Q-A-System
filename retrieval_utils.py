import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_DIR = "data/index"
MODEL_NAME = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR, model_name: str = MODEL_NAME):
        #load FAISS Index
        index_path = os.path.join(index_dir, "kb.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index not found. Run knowledge_base.py first.")
        self.index = faiss.read_index(index_path)

        #Load mapping
        mapping_path = os.path.join(index_dir, "mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)

        #Load embedding model 
        self.model = SentenceTransformer(model_name, device = 'cpu')

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert query string into an embedding vector.
        """
        return self.model.encode([query], convert_to_numpy=True)
    
    def get_top_k(self, query: str, k: int) -> list:
        """
        Retrieve top-K passages relevant to the query.
        Returns a list of dicts: {id, text, score}.
        """
        query_vec = self.embed_query(query)
        D, I = self.index.search(query_vec, k) # D = distances, I = indices

        results = []
        for idx, score in zip(I[0], D[0]):
            passage = self.mapping[str(idx)]
            results.append({
                "id": passage["id"],
                "text": passage["text"],
                "score": float(score) #Lower Score = closer match
            })
        return results
    
# if __name__ == "__main__":
#     retriever = Retriever()
#     query = "What is Sorting?"
#     results = retriever.get_top_k(query, k=2)

#     print(f"Query: {query}\n")
#     for r in results:
#         print(f"[{r['id']}] (score={r['score']:.4f})\n {r['text']}\n")

        

        