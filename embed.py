from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from ingest import extract_text_from_pdf, chunk_text

def build_index(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    faiss.write_index(index, f"{base_name}.faiss")
    with open(f"{base_name}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built for {base_name} with {index.ntotal} vectors")