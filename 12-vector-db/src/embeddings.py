import hashlib
import os
from typing import Any

import redis
import numpy as np
from sentence_transformers import SentenceTransformer

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EMBED_MODEL = "all-MiniLM-L6-v2"
CACHE_TTL = 86_400  # 24h

_model: SentenceTransformer | None = None
_redis: redis.Redis | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=False)
    return _redis


def embed(texts: list[str]) -> tuple[np.ndarray, int, int]:
    """Returns (embeddings, cache_hits, cache_misses)."""
    model = get_model()
    r = get_redis()
    results: list[np.ndarray | None] = [None] * len(texts)
    to_embed: list[tuple[int, str]] = []
    hits = misses = 0

    for i, text in enumerate(texts):
        key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
        cached = r.get(key)
        if cached:
            results[i] = np.frombuffer(cached, dtype=np.float32)
            hits += 1
        else:
            to_embed.append((i, text))
            misses += 1

    if to_embed:
        indices, raw_texts = zip(*to_embed)
        embeddings = model.encode(list(raw_texts), normalize_embeddings=True)
        for idx, emb in zip(indices, embeddings):
            key = f"emb:{hashlib.sha256(texts[idx].encode()).hexdigest()}"
            r.setex(key, CACHE_TTL, emb.astype(np.float32).tobytes())
            results[idx] = emb

    return np.stack(results), hits, misses
