import os
import json
from retrieval_utils import Retriever
from transformers import pipeline, AutoTokenizer

# Configuration
MODEL_NAME = 'microsoft/phi-3-mini-4k-instruct'  # Lightweight, high-performing model (~2.3 GB)
TOP_K = 6  # Number of passages to retrieve (reduced for CPU speed)
MAX_NEW_TOKENS = 600  # Limit generation length
MAX_SEQUENCE_LENGTH = 4096  # Phi-3 supports 4K tokens
TEMPERATURE = 0.7  # For focused output

class RAGQA:
    def __init__(self, retriever: Retriever, generator_model: str = MODEL_NAME):
        self.retriever = retriever
        self.generator = pipeline('text-generation', model=generator_model, device=-1, trust_remote_code=True)  # CPU; trust_remote_code for Phi-3
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model, trust_remote_code=True)  # Load tokenizer

    def format_prompt(self, query: str, passages: list) -> tuple[str, list]:
        """Format the prompt with retrieved passages as context, truncating to fit token limit."""
        context = []
        total_tokens = len(self.tokenizer.encode(f"Question: {query}\nAnswer:"))
        
        for i, passage in enumerate(passages):
            passage_text = f"Passage {i+1}: {passage['text']}"
            passage_tokens = len(self.tokenizer.encode(passage_text))
            if total_tokens + passage_tokens < MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS:  # Reserve space for answer
                context.append(passage_text)
                total_tokens += passage_tokens
            else:
                print(f"Stopped at passage {i+1} due to token limit. Total tokens: {total_tokens}, Passage tokens: {passage_tokens}")
                break

        if not context:
            context = [f"Passage: Limited context available."]
        
        context_text = "\n\n".join(context)
        prompt = f"""Using the following context from MIT OCW lecture transcripts, provide a concise and original answer. Start your response with 'Answer:' and ensure it addresses the question directly. Do not repeat the question or context verbatim.

Context:
{context_text}

Question: {query}

Answer:"""
        print(f"Total tokens used: {total_tokens}, Passages included: {len(context)}")
        return prompt, context

    def generate_answer(self, query: str, k: int = TOP_K) -> dict:
        """Retrieve passages and generate a precise answer based on the context."""
        # Retrieve top-k passages
        retrieved = self.retriever.get_top_k(query, k=k)
        
        if not retrieved:
            return {"answer": "No relevant information found.", "sources": []}
        
        # Format prompt with truncation and get included context
        prompt, context = self.format_prompt(query, retrieved)
        
        # Generate response
        generated = self.generator(prompt, max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=1, 
                                 do_sample=True, temperature=TEMPERATURE, top_k=50, 
                                 pad_token_id=self.tokenizer.eos_token_id, 
                                 eos_token_id=self.tokenizer.eos_token_id)
        answer_text = generated[0]['generated_text'].strip()
        
        # Extract and clean answer
        if "Answer:" in answer_text:
            answer = answer_text.split("Answer:")[-1].strip()
            # Remove any repeated "Question:" sequences
            answer = " ".join(dict.fromkeys(word for word in answer.split() if word != "Question"))
        else:
            answer = "Unable to generate a structured answer based on the context."
        
        # Post-process to ensure conciseness and validity
        words = answer.split()
        answer = " ".join(words[:30]) if words and len(words) > 2 else "No relevant response available."
        
        # Structure the response
        context_count = len(context)
        sources = [{"id": retrieved[i]["id"], "text_snippet": retrieved[i]["text"][:200] + "..." if len(retrieved[i]["text"]) > 200 else retrieved[i]["text"], "score": retrieved[i]["score"]} for i in range(min(context_count, len(retrieved)))]
        
        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(sources)
        }

def main():
    # Initialize Retriever (assumes index is built via knowledge_base.py)
    try:
        retriever = Retriever()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run knowledge_base.py first to build the index.")
        return
    
    # Initialize RAGQA
    rag = RAGQA(retriever)
    
    print("RAG Q&A System for MIT OCW Python Lectures")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        # Generate answer
        response = rag.generate_answer(query)
        
        # Print structured output
        print("\n--- Structured Answer ---")
        print(f"Question: {response['question']}")
        print(f"Answer: {response['answer']}")
        print(f"\nRetrieved {response['retrieval_count']} passages.")
        print("\nSources:")
        for i, source in enumerate(response['sources'], 1):
            print(f"{i}. ID: {source['id']}, Score: {source['score']:.4f}")
            print(f"   Snippet: {source['text_snippet']}\n")

if __name__ == "__main__":
    main()