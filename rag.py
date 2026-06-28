import faiss
import numpy as np
import pickle
import os
import json
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_available_documents():
    return [f.replace(".faiss", "") for f in os.listdir(".") if f.endswith(".faiss")]

def retrieve_from_document(query, doc_name, top_k=3):
    index = faiss.read_index(f"{doc_name}.faiss")
    with open(f"{doc_name}.pkl", "rb") as f:
        chunks = pickle.load(f)
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, doc_names=None):
    if doc_names is None:
        doc_names = get_available_documents()

    all_context = ""
    for doc in doc_names:
        chunks = retrieve_from_document(query, doc)
        all_context += f"\n\n--- From {doc} ---\n" + "\n\n".join(chunks)

    prompt = f"""You are a financial analyst assistant. Use the following excerpts from earnings call transcripts to answer the question. When comparing companies, clearly label which information comes from which company.

Context:
{all_context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_risk_flags(doc_names=None):
    if doc_names is None:
        doc_names = get_available_documents()

    all_flags = {}

    for doc in doc_names:
        with open(f"{doc}.pkl", "rb") as f:
            chunks = pickle.load(f)

        sample_text = "\n\n".join(chunks[:20])

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
