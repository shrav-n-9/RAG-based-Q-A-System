import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PASSAGES_DIR = "data/passages"
INDEX_DIR = "data/index"
MODEL_NAME = 'all-MiniLM-L6-V2'

def load_passages(passages_dir: str = PASSAGES_DIR) -> list:
    #Loads all JSON passage files and returns a list of dicts {id, text}.
    passages = []
    for filename in os.listdir(passages_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(passages_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                passages.extend(data)
    print(f'Loaded {len(passages)} passages')
    return passages

def build_faiss_index(passages: list, model_name: str = MODEL_NAME, out_dir: str = INDEX_DIR):
    #Creates embeddings for passages and stores them in FAISS index. 
    #Saves both the index and the ID -> text mapping
    os.makedirs(out_dir, exist_ok=True)

    #Load Model
    model = SentenceTransformer(model_name)

    #Complete embeddings
    texts = [p["text"] for p in passages]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar = True)

    #FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    #Save Index
    faiss.write_index(index, os.path.join(out_dir, "kb.index"))

    #Save Mapping
    mapping = {i: passages[i] for i in range(len(passages))}
    with open(os.path.join(out_dir, "mapping.json"), "w", encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f'Saved FAISS index and mapping to {out_dir}')

if __name__ == "__main__":
    passages = load_passages()
    build_faiss_index(passages)