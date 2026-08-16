import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .embeddings import embed
from .hybrid_search import build_filter, reciprocal_rank_fusion

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

app = FastAPI(title="Vector DB at Scale")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class CreateCollectionRequest(BaseModel):
    name: str
    vector_size: int = 384
    distance: str = "Cosine"


class IngestRequest(BaseModel):
    collection: str
    documents: list[dict]  # each dict must have "text" and optional metadata keys


class SearchRequest(BaseModel):
    collection: str
    query: str
    k: int = 10
    metadata_filter: dict = {}


@app.post("/collections")
def create_collection(body: CreateCollectionRequest):
    dist = getattr(Distance, body.distance.upper(), Distance.COSINE)
    client.recreate_collection(
        collection_name=body.name,
        vectors_config=VectorParams(size=body.vector_size, distance=dist),
    )
    return {"collection": body.name, "created": True}


@app.post("/ingest")
def ingest(body: IngestRequest):
    texts = [d["text"] for d in body.documents]
    embeddings, hits, misses = embed(texts)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={**doc, "text": doc["text"]},
        )
        for doc, emb in zip(body.documents, embeddings)
    ]
    client.upsert(collection_name=body.collection, points=points)
    return {"ingested": len(points), "cache_hits": hits, "cache_misses": misses}


@app.post("/search")
def search(body: SearchRequest):
    query_emb, _, _ = embed([body.query])
    q_vec = query_emb[0].tolist()

    filt = build_filter(body.metadata_filter)

    # Dense search
    dense = client.search(
        collection_name=body.collection,
        query_vector=q_vec,
        limit=body.k,
        query_filter=filt,
    )

    # RRF fusion (here both lists are dense; plug in sparse encoder for true hybrid)
    fused = reciprocal_rank_fusion(dense, dense)[:body.k]
    return {"results": fused}


@app.get("/collections/{name}/info")
def collection_info(name: str):
    info = client.get_collection(collection_name=name)
    return {"name": name, "points_count": info.points_count, "status": info.status}


@app.post("/collections/{name}/optimize")
def optimize(name: str):
    # Qdrant optimises automatically; this triggers a manual snapshot for demo
    client.update_collection(collection_name=name, optimizer_config={"indexing_threshold": 100})
    return {"collection": name, "optimization": "triggered"}


@app.get("/health")
def health():
    return {"status": "ok"}
