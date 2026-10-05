"""
document_loader.py
-------------------
Ye file 2 kaam karti hai:
1. PDF file se saara text nikalna (extract karna)
2. Us text ko chote-chote overlapping "chunks" mein todhna

Chunking isliye zaroori hai kyunki:
- Poora document ek saath AI ko nahi diya ja sakta (context limit hoti hai)
- Baad mein hum sirf "relevant chunks" dhundhenge, poora document nahi
"""

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Ek PDF file ka path leke, uska saara text ek single string mein return karta hai.

    Example:
        text = extract_text_from_pdf("mydoc.pdf")
    """
    reader = PdfReader(pdf_path)
    full_text = ""

    # Har page ka text nikal ke jodte jao
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Kabhi kabhi page khali bhi ho sakta hai
            full_text += page_text + "\n"

    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Text ko chote-chote chunks mein todhta hai, thoda overlap ke saath.

    chunk_size: har chunk mein kitne words honge (default 500)
    overlap: consecutive chunks ke beech kitne words repeat honge (default 50)

    Example:
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # chunks ek list hogi: ["chunk 1 text...", "chunk 2 text...", ...]
    """
    words = text.split()  # Poore text ko words ki list mein todho
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Agla chunk thoda peeche se shuru hoga (overlap ke liye)
        start += chunk_size - overlap

    return chunks


# ---- Testing ke liye (isko standalone run karke check kar sakte ho) ----
if __name__ == "__main__":
    # Apna test PDF ka path yaha daalo
    test_pdf_path = "sample.pdf"

    print("PDF se text nikal rahe hai...")
    text = extract_text_from_pdf(test_pdf_path)
    print(f"Total characters extracted: {len(text)}")

    print("\nText ko chunks mein todh rahe hai...")
    chunks = chunk_text(text)
    print(f"Total chunks bane: {len(chunks)}")

    print("\n--- Pehla chunk (preview) ---")
    print(chunks[0][:300] + "...")  # Pehle 300 characters dikhao