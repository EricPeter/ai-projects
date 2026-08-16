from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, ScoredPoint


def reciprocal_rank_fusion(
    dense: list[ScoredPoint],
    sparse: list[ScoredPoint],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(dense):
        sid = str(hit.id)
        scores[sid] = scores.get(sid, 0) + 1 / (k + rank + 1)
        docs[sid] = {"id": hit.id, "payload": hit.payload, "score": scores[sid]}

    for rank, hit in enumerate(sparse):
        sid = str(hit.id)
        scores[sid] = scores.get(sid, 0) + 1 / (k + rank + 1)
        if sid not in docs:
            docs[sid] = {"id": hit.id, "payload": hit.payload, "score": scores[sid]}
        else:
            docs[sid]["score"] = scores[sid]

    return sorted(docs.values(), key=lambda x: x["score"], reverse=True)


def build_filter(metadata_query: dict) -> Filter | None:
    if not metadata_query:
        return None
    conditions = []
    for field, condition in metadata_query.items():
        if isinstance(condition, dict):
            if "eq" in condition:
                conditions.append(FieldCondition(key=field, match=MatchValue(value=condition["eq"])))
            elif "gte" in condition or "lte" in condition:
                conditions.append(FieldCondition(
                    key=field,
                    range=Range(gte=condition.get("gte"), lte=condition.get("lte")),
                ))
        else:
            conditions.append(FieldCondition(key=field, match=MatchValue(value=condition)))
    return Filter(must=conditions) if conditions else None
