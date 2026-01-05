import os
import json

PREPROCESSED_DIR = "data/processed"
PASSAGES_DIR = "data/passages"

def split_into_passages(text: str, chunk_size: int=250, overlap: int = 50) -> list:
    """
    Splits text into overlapping passages. 
    Example: 250 words per chunk with 50 words overlap
    """
    words = text.split()
    passages = []

    start = 0
    idx = 1
    while start < len(words):
        end = start + chunk_size 
        chunk = words[start:end]
        if not chunk:
            break
        passages.append(" ".join(chunk))
        start += chunk_size - overlap
        idx += 1

    return passages

def process_text_files(in_dir: str = PREPROCESSED_DIR, out_dir: str = PASSAGES_DIR):
    #Reads all .txt files, splits into passages, and saves as JSON
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(in_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(in_dir, filename)
            lecture_name = os.path.splitext(filename)[0]

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            passages = split_into_passages(text)

            json_data = [
                {"id": f"lecture_name_{i+1}", "text": p}
                for i, p in enumerate(passages)
            ]

            outpath = os.path.join(out_dir, lecture_name + ".json")
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"Processed {filename} -> {outpath}")
if __name__ == "__main__":
    process_text_files()