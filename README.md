# Finance Intelligence Assistant

An AI-powered financial document analysis tool built for the EY Finance Operations value chain. Upload earnings call transcripts and annual reports to query, compare, and identify risks across multiple companies using a Retrieval-Augmented Generation (RAG) pipeline.

## Features

- **Multi-document QA** — Ask questions across multiple uploaded PDFs simultaneously
- **Cross-company comparison** — Compare financials, guidance, and metrics across companies
- **Automated risk flagging** — Detect red flags like revenue misses, margin pressure, and weak guidance
- **Chat interface** — Conversational UI with persistent chat history

## Tech Stack

- **Ingestion** — PyMuPDF for PDF text extraction and chunking
- **Embeddings** — Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store** — FAISS for semantic similarity search
- **LLM** — Groq API (LLaMA 3.1 8B)
- **Backend** — FastAPI
- **Frontend** — Streamlit

## Project Structure

- `ingest.py` — PDF extraction and chunking
- `embed.py` — Embedding generation and FAISS indexing
- `rag.py` — Retrieval and answer generation
- `main.py` — FastAPI backend
- `app.py` — Streamlit frontend
- `.streamlit/config.toml` — UI theme configuration 

## Setup

1. Clone the repo
2. Install dependencies:
```bash
   pip install pymupdf sentence-transformers faiss-cpu groq fastapi uvicorn python-multipart streamlit python-dotenv
```
3. Create a `.env` file:
GROQ_API_KEY=your-groq-api-key
4. Run the backend:
```bash
   uvicorn main:app --reload
```
5. Run the frontend:
```bash
   streamlit run app.py
```

## Usage

1. Upload one or more earnings call transcript PDFs via the sidebar
2. Ask questions in the chat — the system retrieves relevant chunks and generates grounded answers
3. Click **Generate Risk Flags** to automatically surface financial red flags across all documents

## Sample Queries

- "What was TCS revenue this quarter?"
- "Compare TCS and Infosys revenue growth"
- "What are the key risks mentioned by Infosys management?"
- "What is the operating margin guidance for next year?"