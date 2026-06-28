import faiss
import numpy as np
import pickle
import os
import google.generativeai as genai
from ingest import extract_text_from_pdf, chunk_text
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        embeddings.append(result['embedding'])
    return embeddings

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
