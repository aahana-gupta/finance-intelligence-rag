import faiss
import numpy as np
import pickle
import os
import requests
from ingest import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings(texts):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    return response.json()

def build_index(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    embeddings = get_embeddings(chunks)
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    faiss.write_index(index, f"{base_name}.faiss")
    with open(f"{base_name}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built for {base_name} with {index.ntotal} vectors")