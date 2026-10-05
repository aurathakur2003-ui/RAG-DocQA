"""
vector_store.py
----------------
Ye file 2 kaam karti hai:
1. Text chunks ko "embeddings" (numbers ke vectors) mein convert karna
2. Un embeddings ko ChromaDB mein store karna, taaki baad mein
   "semantic search" kiya ja sake (matlab meaning ke basis pe search,
   sirf keyword match nahi)

Simple example:
   "cat" aur "kitten" alag words hai, lekin meaning mein similar hai.
   Embeddings isi similarity ko numbers mein capture karte hai.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# Ye model text ko embeddings mein convert karega.
# Pehli baar chalane pe ye model download hoga (thoda time lagega),
# uske baad local mein hi save rahega.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_chroma_collection(collection_name="documents"):
    """
    ChromaDB ka ek "collection" banata ya kholta hai - isse hum
    ek table/folder samajh sakte hai jaha saare chunks store honge.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def store_chunks(chunks, collection_name="documents"):
    """
    Chunks ki list leke, unke embeddings banata hai aur ChromaDB mein store karta hai.

    chunks: list of text chunks (document_loader.py se aaye hue)

    Example:
        store_chunks(chunks)
    """
    collection = get_chroma_collection(collection_name)

    # Har chunk ke liye embedding banao
    print(f"{len(chunks)} chunks ke embeddings ban rahe hai...")
    embeddings = embedding_model.encode(chunks).tolist()

    # Har chunk ko ek unique ID chahiye ChromaDB mein
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # ChromaDB mein store karo - text, embedding, aur id teeno saath
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print(f"{len(chunks)} chunks ChromaDB mein store ho gaye.")


def search_relevant_chunks(query, collection_name="documents", top_k=3):
    """
    User ke question (query) ke liye, sabse relevant chunks dhundhta hai.

    query: user ka question (jaise "Data visualization kya hai?")
    top_k: kitne sabse relevant chunks chahiye (default 3)

    Returns: list of relevant text chunks
    """
    collection = get_chroma_collection(collection_name)

    # Query ka bhi embedding banao (usi model se, taaki compare ho sake)
    query_embedding = embedding_model.encode([query]).tolist()

    # ChromaDB se sabse similar chunks maango
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # results['documents'][0] mein relevant chunks ki list hoti hai
    relevant_chunks = results['documents'][0]
    return relevant_chunks


# ---- Testing ke liye ----
if __name__ == "__main__":
    from document_loader import extract_text_from_pdf, chunk_text

    # Pehle document load aur chunk karo
    print("PDF se text nikal rahe hai...")
    text = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(text)

    # Chunks ko store karo
    store_chunks(chunks)

    # Ab ek test query karke dekho
    test_query = "What is data analytics?"
    print(f"\nTest query: '{test_query}'")
    print("Relevant chunks dhundh rahe hai...\n")

    relevant = search_relevant_chunks(test_query)
    for i, chunk in enumerate(relevant):
        print(f"--- Relevant Chunk {i+1} ---")
        print(chunk[:200] + "...\n")