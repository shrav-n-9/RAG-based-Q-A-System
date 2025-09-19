# evaluate_rag.py

import json
import argparse
from sentence_transformers import SentenceTransformer, util
from retrieval_utils import Retriever
from rag import RAGQA   # <-- uses your existing local rag.py pipeline

# Load semantic similarity model once
sim_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_gold(path: str):
    """
    Load gold dataset with:
    - question
    - answer (gold)
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_answer(gold_answer: str, generated_answer: str) -> float:
    """
    Compute semantic similarity between gold and generated answers.
    Returns cosine similarity in [0,1].
    """
    emb_gold = sim_model.encode(gold_answer, convert_to_tensor=True)
    emb_gen = sim_model.encode(generated_answer, convert_to_tensor=True)
    sim = util.cos_sim(emb_gold, emb_gen).item()
    return sim

def evaluate_rag(gold_data, ragqa: RAGQA, k: int = 3):
    results = []
    sims = []

    for item in gold_data:
        q = item["question"]
        gold_ans = item["answer"]

        # Run your local RAG pipeline
        result = ragqa.generate_answer(q, k=k)
        gen_ans = result["answer"]

        sim = score_answer(gold_ans, gen_ans)
        sims.append(sim)

        results.append({
            "question": q,
            "gold_answer": gold_ans,
            "generated_answer": gen_ans,
            "similarity": round(sim, 3),
            "retrieved_ids": [s["id"] for s in result["sources"]]
        })

    avg_sim = sum(sims) / len(sims) if sims else 0.0
    return avg_sim, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate local RAG (llamafile) answers vs gold dataset")
    parser.add_argument("gold_file", type=str, help="Path to gold Q&A dataset (JSON).")
    parser.add_argument("--k", type=int, default=3, help="Top-k passages to retrieve.")
    args = parser.parse_args()

    # Init retriever + RAG pipeline
    retriever = Retriever()
    ragqa = RAGQA(retriever)

    # Load gold dataset
    gold_data = load_gold(args.gold_file)

    # Run evaluation
    avg_sim, results = evaluate_rag(gold_data, ragqa, k=args.k)

    print(f"\n📊 Average Semantic Similarity: {avg_sim:.3f}\n")

    for r in results:
        print(f"Q: {r['question']}")
        print(f"Gold: {r['gold_answer']}")
        print(f"Gen : {r['generated_answer']}")
        print(f"Sim : {r['similarity']}")
        print(f"Retrieved IDs: {r['retrieved_ids']}")
        print("---")

if __name__ == "__main__":
    main()
