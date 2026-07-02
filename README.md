# Finance Intelligence Assistant

An AI-powered financial document analysis tool built as part of an EY internship project on **"Use of GenAI in the Finance Operations Value Chain"**. Upload earnings call transcripts and annual reports to query, compare, and identify risks across multiple companies using a Retrieval-Augmented Generation (RAG) pipeline.

## Demo

![Risk Flags](demo1.webp)
![QA Interface](demo2.webp)

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

## Deployment

- **Frontend:** Deployed on Render at [https://finance-intelligence-rag-frontend.onrender.com](https://finance-intelligence-rag-frontend.onrender.com) — note this requires the backend running locally to function
- **Backend:** Runs locally due to memory constraints on free hosting tiers (Sentence Transformers requires ~400MB RAM, exceeding Render's 512MB free plan limit)

**To run the full app:**
1. Start the backend: `uvicorn main:app --reload`
2. Start the frontend: `streamlit run app.py`
3. Open `http://localhost:8501` in your browser

## Setup

1. Clone the repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

## Security

- **Prompt injection detection** — Uploaded documents are scanned before indexing for patterns like "ignore previous instructions", "system prompt", "jailbreak", and "disregard"; matches raise a `ValueError` and abort indexing
- **Grounded generation** — The LLM prompt restricts answers to the retrieved context and instructs the model to say so when the answer isn't present, reducing hallucination and prompt leakage
- **Filename sanitization** — Uploaded filenames are stripped to their base name (`os.path.basename`) to prevent path traversal on write
- **File type validation** — Only `.pdf` uploads are accepted; all other file types are rejected
- **Rate limiting** — `/ask` and `/upload` are capped at 10 requests/minute per client via `slowapi` to guard against abuse
- **Source attribution** — Every generated answer is appended with the source document names for traceability and auditability

## Testing

Run the test suite with:
```bash
pytest tests/
```

The tests cover:
- **Text extraction** — `extract_text_from_pdf` returns non-empty text from a sample PDF
- **Chunking** — `chunk_text` splits long text into multiple correctly-sized chunks
- **Prompt injection detection** — flagged text raises `ValueError`, clean financial text does not
- **Document listing** — `get_available_documents` returns a list
- **Full pipeline (integration)** — uploading and indexing a sample PDF followed by `generate_answer` returns a non-empty answer end to end

Planned: a RAGAS-based evaluation (faithfulness, answer relevancy, context precision) once a ground-truth QA dataset is available.