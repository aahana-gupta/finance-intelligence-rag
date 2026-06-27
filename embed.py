import faiss
import numpy as np
import pickle
import os
from openai import OpenAI
from ingest import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

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