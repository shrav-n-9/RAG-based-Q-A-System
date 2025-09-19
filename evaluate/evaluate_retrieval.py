# evaluate_retrieval.py

import json
import argparse
from retrieval_utils import Retriever

def load_gold(path: str):
    """
    Load gold Q&A dataset.
    Expected format: list of dicts with keys:
    - "question": str
    - "answer": str
    - "relevant_ids": list of passage IDs (from bootstrap or manual curation)
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_retrieval(gold_data, retriever, k: int = 3):
    """
    For each gold question, retrieve top-k passages and check if
    any relevant_ids appear in retrieved results.
    Returns recall@k and detailed results.
    """
    total = len(gold_data)
    hits = 0
    detailed_results = []

    for item in gold_data:
        q = item["question"]
        relevant_ids = set(item.get("relevant_ids", []))
        retrieved = retriever.get_top_k(q, k=k)
        retrieved_ids = {r["id"] for r in retrieved}

        hit = len(relevant_ids & retrieved_ids) > 0
        if hit:
            hits += 1

        detailed_results.append({
            "question": q,
            "relevant_ids": list(relevant_ids),
            "retrieved_ids": list(retrieved_ids),
            "hit": hit
        })

    recall = hits / total if total > 0 else 0.0
    return recall, detailed_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval using gold dataset")
    parser.add_argument("gold_file", type=str, help="Path to gold Q&A dataset (JSON).")
    parser.add_argument("--k", type=int, default=3, help="Top-k passages to retrieve.")
    args = parser.parse_args()

    # Load dataset
    gold_data = load_gold(args.gold_file)

    # Init retriever
    retriever = Retriever()

    # Run evaluation
    recall, details = evaluate_retrieval(gold_data, retriever, k=args.k)

    print(f"\n📊 Recall@{args.k}: {recall:.2f} ({int(recall*100)}%)\n")

    for d in details:
        print(f"Q: {d['question']}")
        print(f"Relevant: {d['relevant_ids']}")
        print(f"Retrieved: {d['retrieved_ids']}")
        print(f"Hit: {d['hit']}")
        print("---")

if __name__ == "__main__":
    main()
