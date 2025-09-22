import os
import json
import subprocess
from retrieval_utils import Retriever

# Configuration
MODEL_FILE = 'llama-3-8b-instruct-q4.gguf'  
LLAMA_FILE_EXE = 'llamafile-0.9.3.exe'  
TOP_K = 4  
MAX_NEW_TOKENS = 200 
TEMPERATURE = 0.5  

class RAGQA:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        if not os.path.exists(LLAMA_FILE_EXE):
            raise FileNotFoundError(f"{LLAMA_FILE_EXE} not found. Download from https://github.com/Mozilla-Ocho/llamafile/releases and rename to llamafile.exe.")
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"{MODEL_FILE} not found. Download with: huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir . and rename to {MODEL_FILE}.")

    def format_prompt(self, query: str, passages: list) -> str:
        """Format the prompt with shortened retrieved passages as context to reduce length."""
        context_text = "\n\n".join([f"Passage {i+1}: {p['text'][:200]}..." for i, p in enumerate(passages)])  # Shorten each passage to 200 characters
        prompt = f"""Using the following context from MIT OCW lecture transcripts, provide a concise answer in bullet points (max 4-5 points) that is crisp and understandable, fully addressing the question. Start with 'Answer:' followed by the bullet points prefixed with '- ' (e.g., - 1st Point, - 2nd Point). Each bullet point in a new line. Synthesize key insights from all provided passages without repeating the question or context verbatim. If relevant information is insufficient, include a bullet explaining the limitation.Ensure the Answer is complete with no loose ending.

Context:
{context_text}

Question: {query}

Answer:"""
        print(f"Passages included: {len(passages)}")
        return prompt

    def generate_answer(self, query: str, k: int = TOP_K) -> dict:
        """Retrieve passages and generate a precise paragraph-sized answer using llamafile."""
        retrieved = self.retriever.get_top_k(query, k=k)
        
        if not retrieved:
            return {"answer": "No relevant information found.", "sources": []}
        
        prompt = self.format_prompt(query, retrieved)
        
        cmd = [
            LLAMA_FILE_EXE,
            '--cli',  
            '-m', MODEL_FILE,  
            '--prompt', prompt,
            '--n-predict', str(MAX_NEW_TOKENS),
            '--temp', str(TEMPERATURE),
            '--top-k', '50',
            '--no-display-prompt',
            '--threads', '4'  
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)  
        
        if result.returncode != 0:
            print(f"Generation error (return code {result.returncode}): {result.stderr}")
            answer = "Generation failed. Check model file and llamafile parameters."
        else:
            answer = result.stdout.strip().split("Answer:")[-1].strip() #if "Answer:" in result.stdout else result.stdout.strip()
            lines = [line.strip() for line in answer.split('\n') if line.strip() and not line.strip().startswith("Answer:")]
            answer = '\n'.join(lines) if lines else "No response generated."
        
        words = answer.split()
        answer = '\n'.join(answer.split('\n')[:5]) if len(words) > 20 else answer if len(words) > 2 else "The retrieved context lacks sufficient relevant information to generate a detailed response."

        sources = [{"id": p["id"], "text_snippet": p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"], "score": p["score"]} for p in retrieved]
        
        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(sources)
        }

# def main():
#     try:
#         retriever = Retriever()
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Run knowledge_base.py first to build the index.")
#         return
    
#     rag = RAGQA(retriever)
    
#     print("RAG Q&A System for MIT OCW Python Lectures (Powered by LlamaFile with Llama)")
#     print("Type 'quit' to exit.\n")
    
#     while True:
#         query = input("Enter your question: ").strip()
#         if query.lower() == 'quit':
#             break
        
#         if not query:
#             continue
        
#         response = rag.generate_answer(query)
        
#         print("\n--- Structured Answer ---")
#         print(f"Question: {response['question']}")
#         print(f"Answer: {response['answer']}")
#         print(f"\nRetrieved {response['retrieval_count']} passages.")
#         print("\nSources:")
#         for i, source in enumerate(response['sources'], 1):
#             print(f"{i}. ID: {source['id']}, Score: {source['score']:.4f}")
#             print(f"   Snippet: {source['text_snippet']}\n")

# if __name__ == "__main__":
#     main()