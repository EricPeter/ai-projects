from .ingestion import get_model, _get_table


def search(query: str, k: int = 5) -> list[dict]:
    model = get_model()
    q_emb = model.encode(query, normalize_embeddings=True).tolist()
    table = _get_table()
    results = table.search(q_emb).limit(k).to_list()
    return [{"content": r["content"], "source": r["source"], "score": round(1 - r.get("_distance", 0), 4)} for r in results]
