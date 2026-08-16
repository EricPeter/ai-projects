import json
import struct
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .ingestion import get_db, get_embed_model

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10
RERANK_TOP_K = 5

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def _dense_search(query_embedding: list[float], k: int) -> list[dict[str, Any]]:
    conn = get_db()
    vec_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    rows = conn.execute(
        """
        SELECT c.id, c.page_number, c.content,
               vec_distance_cosine(cv.embedding, ?) AS distance
        FROM chunk_vectors cv
        JOIN chunks c ON c.id = cv.chunk_id
        ORDER BY distance ASC
        LIMIT ?
        """,
        (vec_bytes, k),
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "chunk_id": row[0],
            "page_number": row[1],
            "content": row[2],
            "dense_score": 1 - row[3],  # convert distance to similarity
        })
    return results


def _bm25_search(query: str, k: int) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi

    conn = get_db()
    rows = conn.execute("SELECT doc_id, index_data FROM bm25_index").fetchall()
    conn.close()

    if not rows:
        return []

    query_tokens = query.lower().split()
    all_results: list[dict[str, Any]] = []

    for _, index_data_raw in rows:
        idx = json.loads(index_data_raw)
        corpus = idx["corpus"]
        pages = idx["pages"]
        ids = idx["ids"]

        tokenized = [t.lower().split() for t in corpus]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]
        for i in top_indices:
            if scores[i] > 0:
                all_results.append({
                    "chunk_id": ids[i],
                    "page_number": pages[i],
                    "content": corpus[i],
                    "bm25_score": float(scores[i]),
                })

    all_results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return all_results[:k]


def _reciprocal_rank_fusion(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}

    for rank, doc in enumerate(dense):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        docs[cid] = doc

    for rank, doc in enumerate(sparse):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        if cid not in docs:
            docs[cid] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{"rrf_score": scores[cid], **docs[cid]} for cid in sorted_ids]


def hybrid_search_and_rerank(query: str, top_k: int = RERANK_TOP_K) -> list[dict[str, Any]]:
    model = get_embed_model()
    query_emb = model.encode(query, normalize_embeddings=True).tolist()

    dense_results = _dense_search(query_emb, TOP_K)
    sparse_results = _bm25_search(query, TOP_K)

    fused = _reciprocal_rank_fusion(dense_results, sparse_results)[:TOP_K]

    if not fused:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(query, doc["content"]) for doc in fused]
    ce_scores = cross_encoder.predict(pairs)

    for doc, score in zip(fused, ce_scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(fused, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    return reranked
