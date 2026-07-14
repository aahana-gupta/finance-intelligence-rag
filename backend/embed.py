import numpy as np
import os
import uuid
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from backend.ingest import extract_text_from_pdf, chunk_text
from backend.rag import check_prompt_injection
from backend.db import client, ensure_collection, COLLECTION_NAME, embedding_model as model

def build_index(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    check_prompt_injection(text)
    chunks = chunk_text(text)

    # fastembed's .embed() returns a generator of numpy arrays (not a single batched call like .encode()).
    # batch_size caps how many chunks the ONNX runtime processes per inference call - the default (256)
    # processes an entire document's chunks in one batch, spiking memory well past Render's 512MB limit.
    embeddings_np = np.array(list(model.embed(chunks, batch_size=8))).astype("float32")

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    ensure_collection()

    # Remove any existing vectors for this document before re-uploading, so
    # re-running build_index() on the same file doesn't silently duplicate
    # vectors in the collection (this bit us during eval.py testing earlier).
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=base_name))]
        ),
    )

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

