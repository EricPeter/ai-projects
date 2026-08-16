"""Long-term memory using Qdrant. Stores extracted facts per user."""
import os
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = "long_term_memory"
DIM = 384
MAX_MEMORIES_PER_USER = 1_000
EVICTION_AGE_DAYS = 30

_client: QdrantClient | None = None
_model: SentenceTransformer | None = None


def _q() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        try:
            _client.get_collection(COLLECTION)
        except Exception:
            _client.recreate_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    return _client


def _m() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def store(user_id: str, fact: str) -> None:
    emb = _m().encode(fact, normalize_embeddings=True).tolist()
    _q().upsert(COLLECTION, points=[
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"user_id": user_id, "fact": fact, "created_at": datetime.now(timezone.utc).isoformat()},
        )
    ])


def recall(user_id: str, query: str, k: int = 5) -> list[str]:
    q_emb = _m().encode(query, normalize_embeddings=True).tolist()
    hits = _q().search(
        collection_name=COLLECTION,
        query_vector=q_emb,
        limit=k,
        query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
    )
    return [h.payload["fact"] for h in hits]


def get_all(user_id: str) -> list[dict]:
    result = _q().scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=100,
    )
    return [{"fact": p.payload["fact"], "created_at": p.payload["created_at"]} for p in result[0]]
