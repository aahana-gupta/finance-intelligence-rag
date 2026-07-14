import os
import json
from groq import Groq
from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue
from db import client, COLLECTION_NAME, embedding_model as model
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "jailbreak",
    "disregard",
    "forget previous",
]

def check_prompt_injection(text):
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lower:
            raise ValueError(f"Potential prompt injection detected in document: '{pattern}'")

def get_available_documents():
    """Returns the distinct doc_name values currently stored in the Qdrant collection."""
    if not client.collection_exists(COLLECTION_NAME):
        return []
    doc_names = set()
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=200,
            offset=next_offset,
        )
        for r in records:
            doc_names.add(r.payload["doc_name"])
        if next_offset is None:
            break
    return sorted(doc_names)

def retrieve_from_document(query, doc_name, top_k=3):
    query_embedding = list(model.embed([query]))[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
        ),
        limit=top_k,
    )
    return [point.payload["text"] for point in results.points]

def get_chunks_for_document(doc_name, limit=20):
    """Returns the first `limit` chunks (by original order) for a document, used for risk analysis."""
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=1000,
    )
    records_sorted = sorted(records, key=lambda r: r.payload.get("chunk_index", 0))
    return [r.payload["text"] for r in records_sorted[:limit]]

def generate_answer(query, doc_names=None):
    if doc_names is None:
        doc_names = get_available_documents()

    all_context = ""
    retrieved_chunks = []  # flat list of raw retrieved chunk text, needed for eval (e.g. Ragas contexts)
    for doc in doc_names:
        chunks = retrieve_from_document(query, doc)
        retrieved_chunks.extend(chunks)
        all_context += f"\n\n--- From {doc} ---\n" + "\n\n".join(chunks)

    prompt = f"""You are a financial analyst assistant. Use the following excerpts from earnings call transcripts to answer the question. When comparing companies, clearly label which information comes from which company. Only use the provided context. If the answer is not in the context, say: I don't have enough information to answer this based on the provided documents.

Context:
{all_context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_answer = response.choices[0].message.content
    sources = ", ".join(doc_names)

    return {
        "answer": f"{raw_answer}\n\nSources: {sources}",  # keeps existing UI behavior (Streamlit displays this as-is)
        "raw_answer": raw_answer,      # answer text without the appended sources line, useful for eval
        "contexts": retrieved_chunks,  # clean list of retrieved chunk strings, required by Ragas-style evals
        "sources": doc_names,
    }

def generate_risk_flags(doc_names=None):
    if doc_names is None:
        doc_names = get_available_documents()

    all_flags = {}

    for doc in doc_names:
        chunks = get_chunks_for_document(doc, limit=20)
        sample_text = "\n\n".join(chunks)

        prompt = f"""You are a financial risk analyst. Analyze the following earnings call transcript excerpt and identify red flags or risks.

Look for: revenue misses, margin pressure, declining growth, client losses, weak guidance, macroeconomic concerns, litigation, leadership changes, or any cautionary language.

Return a JSON array of objects with keys "flag" (short title) and "detail" (one sentence explanation). Return only JSON, no other text.

Transcript:
{sample_text}"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            flags = json.loads(response.choices[0].message.content)
        except:
            flags = [{"flag": "Parse error", "detail": "Could not extract flags"}]

        all_flags[doc] = flags

    return all_flags
