import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from fastembed import TextEmbedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "finance_documents"
VECTOR_SIZE = 384  # must match sentence-transformers/all-MiniLM-L6-v2's output dimension

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Loaded once here, imported everywhere else that needs embeddings.
# Uses fastembed (ONNX Runtime) instead of sentence-transformers (PyTorch) -
# same model, same 384-dim output, but without pulling in torch/CUDA, which
# was the dominant cause of OOM crashes on Render's 512MB free tier.
embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ensure_collection():
    """Creates the collection if it doesn't already exist. Safe to call every time."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="doc_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )
