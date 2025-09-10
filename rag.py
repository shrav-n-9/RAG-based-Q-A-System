import os
from dotenv import load_dotenv
from openai import OpenAI
from retrieval_utils import Retriever

#Load OpenAI key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def rag_query(user_query: str, k:int = 3, model: str = "gpt-3.5-turbo") -> str:
    """
    Runs a RAG pipeline:
    1. Retrieve top-k passages from FAISS
    2. Pass them + query into OpenAI model
    3. Return generated answer
    """
    retriever = Retriever()
    results = retriever.get_top_k(user_query, k=k)

    #Combine retrieved passages into context
    context = "\n\n".join([r["text"] for r in results])

    #Build prompt
    prompt = f"""
    You are a helful assistant answering based on lecture transcripts.
    Use the context below to answer the questions concisely. 

    Context:
    {context}

    Question:
    {user_query}

    Answer:
    """
    #Call OpenAPI
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content": prompt}],
        temperature = 0.2
    )

    return response.choices[0].message.content.strip()

# if __name__ == "__main__":
#     query = "Define Python Programming."
#     answer = rag_query(query, k=2)
#     print(f"Q: {query}\n A: {answer}")

