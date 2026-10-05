"""
rag_engine.py
-------------
Ye file RAG pipeline ka "brain" hai. Isme 2 kaam hote hai:

1. User ke question ke liye relevant chunks dhundhna (vector_store.py use karke)
2. Un chunks ko "context" ke roop mein local LLM (Ollama) ko dena, taaki
   model sirf apne training data se guess na kare, balki tumhare document
   ke actual content ke basis par answer de.

Isi combination (Retrieval + Generation) ko RAG kehte hai.

NOTE: Ye version Ollama (free, local LLM) use karta hai - koi API key
ya internet cost nahi lagti. Model tumhare khud ke computer pe chalta hai.
Isse pehle 'ollama pull llama3.2:1b' chala ke model download kar lena,
aur Ollama app background mein chal rahi honi chahiye.
"""

import ollama
from vector_store import search_relevant_chunks

# Konsa local model use karna hai
OLLAMA_MODEL = "llama3.2:1b"


def build_prompt(query, relevant_chunks):
    """
    Relevant chunks aur user ke question ko milake ek prompt banata hai,
    jo LLM ko bheja jayega.

    Isme hum model ko clearly bolte hai ki sirf diye gaye context
    se answer de, apni taraf se information mat jodo.
    """
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"""Use the following context to answer the question. Base your answer only on the context below.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def get_answer(query, top_k=3):
    """
    Ye main function hai jo poora RAG flow chalata hai:
    1. Query ke liye relevant chunks dhundo
    2. Prompt banao (chunks + query milake)
    3. Ollama (local LLM) ko call karo
    4. Answer wapas do

    Example:
        answer = get_answer("Data analytics ke types kya hai?")
        print(answer)
    """
    # Step 1: Relevant chunks dhundo
    relevant_chunks = search_relevant_chunks(query, top_k=top_k)

    # Step 2: Prompt banao
    prompt = build_prompt(query, relevant_chunks)

    # Step 3: Ollama (local model) ko call karo
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Step 4: Answer nikal ke return karo
    answer = response["message"]["content"]
    return answer


# ---- Testing ke liye ----
if __name__ == "__main__":
    test_query = "What is data analytics?"

    print(f"Question: {test_query}\n")
    print("Answer dhundh rahe hai (relevant chunks + Ollama local LLM)...\n")

    answer = get_answer(test_query)

    print("--- Ollama ka Answer ---")
    print(answer)