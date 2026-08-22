import json
import time
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"
COLLECTION = "test_col_pytest"


def _baseline(name, current):
    BASELINES.mkdir(exist_ok=True)
    p = BASELINES / f"{name}.json"
    if not p.exists():
        p.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline '{name}' created")
    return json.loads(p.read_text())


@pytest.fixture(autouse=True)
def fresh_collection(base_url):
    httpx.delete(f"{base_url}/collections/{COLLECTION}", timeout=10)
    httpx.post(
        f"{base_url}/collections",
        json={"name": COLLECTION, "vector_size": 384},
        timeout=10,
    )
    yield
    httpx.delete(f"{base_url}/collections/{COLLECTION}", timeout=10)


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_search(base_url, msmarco_sample):
    docs = [{"text": row["text"]} for row in msmarco_sample]
    r = httpx.post(
        f"{base_url}/ingest",
        json={"collection": COLLECTION, "documents": docs},
        timeout=30,
    )
    assert r.status_code == 200
    assert r.json().get("inserted", 0) == len(docs)

    r = httpx.post(
        f"{base_url}/search",
        json={"collection": COLLECTION, "query": msmarco_sample[0]["query"], "top_k": 3},
        timeout=30,
    )
    assert r.status_code == 200
    results = r.json().get("results", [])
    assert len(results) <= 3


def test_search_empty_collection_returns_empty(base_url):
    r = httpx.post(
        f"{base_url}/search",
        json={"collection": COLLECTION, "query": "test query", "top_k": 3},
        timeout=10,
    )
    assert r.status_code == 200
    assert r.json().get("results", []) == []


# ── Regression ────────────────────────────────────────────────────────────────

def test_search_result_schema_stable(base_url, msmarco_sample):
    docs = [{"text": row["text"]} for row in msmarco_sample[:2]]
    httpx.post(f"{base_url}/ingest", json={"collection": COLLECTION, "documents": docs}, timeout=30)

    r = httpx.post(
        f"{base_url}/search",
        json={"collection": COLLECTION, "query": msmarco_sample[0]["query"], "top_k": 1},
        timeout=10,
    )
    results = r.json().get("results", [])
    if results:
        keys = sorted(results[0].keys())
        stored = _baseline("result_keys", {"keys": keys})
        assert keys == stored["keys"]


# ── A/B ── top_k=3 vs top_k=10 ───────────────────────────────────────────────

def test_top_k_comparison(base_url, msmarco_sample):
    docs = [{"text": row["text"]} for row in msmarco_sample]
    httpx.post(f"{base_url}/ingest", json={"collection": COLLECTION, "documents": docs}, timeout=30)

    query = msmarco_sample[0]["query"]

    t0 = time.perf_counter()
    r3 = httpx.post(f"{base_url}/search", json={"collection": COLLECTION, "query": query, "top_k": 3}, timeout=10)
    ms3 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r10 = httpx.post(f"{base_url}/search", json={"collection": COLLECTION, "query": query, "top_k": 10}, timeout=10)
    ms10 = (time.perf_counter() - t0) * 1000

    print(f"\n  top_k=3:  {len(r3.json()['results'])} results  {ms3:.0f}ms")
    print(f"  top_k=10: {len(r10.json()['results'])} results  {ms10:.0f}ms")

    assert len(r3.json()["results"]) <= 3
    assert len(r10.json()["results"]) <= 10
