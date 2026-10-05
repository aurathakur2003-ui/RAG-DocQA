# RAG Document Q&A System

An AI-powered **Retrieval-Augmented Generation (RAG)** system that lets users upload a PDF document and ask natural language questions about it. Answers are generated strictly from the document's content — not from the model's general knowledge — making responses accurate and grounded.

## What is RAG?

RAG combines two steps:

1. **Retrieval** — The document is split into chunks, converted into vector embeddings, and stored in a vector database. When a question is asked, the most semantically relevant chunks are retrieved (based on meaning, not just keyword matching).
2. **Generation** — The retrieved chunks are passed to a language model along with the question, which then generates a natural, accurate answer grounded in the actual document content.

## Features

- **PDF Upload** — Upload any PDF document through a simple web interface.
- **Semantic Search** — Finds relevant content based on meaning, not just keywords, using sentence embeddings.
- **Local LLM Inference** — Runs entirely on a local model via Ollama, with no API costs or external dependencies.
- **Grounded Answers** — The model is instructed to answer only from the retrieved context, reducing hallucination.
- **Simple Web UI** — Built with Streamlit for an easy upload-and-ask experience.

## Project Structure

```
RAG-DocQA/
├── document_loader.py   # Extracts text from PDF and splits it into overlapping chunks
├── vector_store.py       # Generates embeddings and stores/retrieves them via ChromaDB
├── rag_engine.py          # Combines retrieval with local LLM generation
├── app.py                 # Streamlit web interface
├── requirements.txt       # Project dependencies
└── .env                    # Environment variables (not committed)
```

## How It Works

1. User uploads a PDF through the Streamlit interface.
2. The document is parsed and split into ~500-word overlapping chunks.
3. Each chunk is converted into an embedding using `sentence-transformers` and stored in ChromaDB.
4. When a question is submitted, its embedding is compared against stored chunks to retrieve the top-k most relevant ones.
5. The relevant chunks and the question are passed into a prompt sent to a local LLM (via Ollama).
6. The model generates an answer based only on the retrieved context, which is displayed in the UI.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM Inference | Ollama (local, free — Llama 3.2) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| PDF Parsing | PyPDF |
| Web Interface | Streamlit |

## How to Run

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed, with a model pulled:
  ```bash
  ollama pull llama3.2:1b
  ```

### Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`), upload a PDF, and start asking questions.

## Notes

- This project originally supported the Anthropic Claude API as the generation backend; it was switched to a local Ollama model to keep the project fully free to run and self-contained, with no external API dependency.
- The code is modular — swapping the LLM backend (e.g., back to Claude, or to another provider) only requires changing `rag_engine.py`.

## Author

Gaurav Singh — MCA (AI/ML), Chandigarh University