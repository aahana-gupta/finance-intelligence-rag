from sentence_transformers import SentenceTransformer
import numpy as np
import os
import uuid
from qdrant_client.models import PointStruct
from ingest import extract_text_from_pdf, chunk_text
from rag import check_prompt_injection
from db import client, ensure_collection, COLLECTION_NAME

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    check_prompt_injection(text)
    chunks = chunk_text(text)

    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings_np = np.array(embeddings).astype("float32")

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    ensure_collection()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings_np[i].tolist(),
            payload={"text": chunks[i], "doc_name": base_name, "chunk_index": i},
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Index built for {base_name} with {len(points)} vectors uploaded to Qdrant")

