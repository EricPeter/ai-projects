import time
import anthropic
from sentence_transformers import SentenceTransformer
from .store import save_snapshot
from .probes import Probe

_client: anthropic.AsyncAnthropic | None = None
_embedder: SentenceTransformer | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


async def run_probe(model: str, probe: Probe) -> dict:
    client = _get_client()
    embedder = _get_embedder()

    start = time.perf_counter()
    message = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": probe.prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    response = message.content[0].text
    tokens = message.usage.input_tokens + message.usage.output_tokens
    embedding = embedder.encode(response).tolist()

    await save_snapshot(model, probe.id, response, embedding, latency_ms, tokens)

    keyword_hits = [kw for kw in probe.expected_keywords if kw.lower() in response.lower()]
    return {
        "model": model,
        "probe_id": probe.id,
        "response": response,
        "latency_ms": round(latency_ms, 2),
        "tokens": tokens,
        "keyword_hits": keyword_hits,
        "keyword_score": len(keyword_hits) / len(probe.expected_keywords) if probe.expected_keywords else 1.0,
    }
