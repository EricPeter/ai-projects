from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_st: SentenceTransformer | None = None


def _get_st():
    global _st
    if _st is None:
        _st = SentenceTransformer("all-MiniLM-L6-v2")
    return _st


def score(prediction: str, reference: str) -> dict[str, float]:
    rouge_l = _rouge.score(reference, prediction)["rougeL"].fmeasure
    embs = _get_st().encode([prediction, reference], normalize_embeddings=True)
    semantic_sim = float(np.dot(embs[0], embs[1]))
    return {"rouge_l": round(rouge_l, 4), "semantic_similarity": round(semantic_sim, 4)}
