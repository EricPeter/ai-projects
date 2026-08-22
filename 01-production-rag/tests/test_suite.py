import json
import time
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"


def _baseline(name, current):
    BASELINES.mkdir(exist_ok=True)
    p = BASELINES / f"{name}.json"
    if not p.exists():
        p.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline '{name}' created")
    return json.loads(p.read_text())


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_returns_answer_and_sources(base_url, squad_sample):
    for row in squad_sample[:2]:
        r = httpx.post(
            f"{base_url}/query",
            json={"question": row["question"]},
            timeout=30,
        )
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert isinstance(body["sources"], list)


def test_query_empty_question_rejected(base_url):
    r = httpx.post(f"{base_url}/query", json={"question": ""}, timeout=10)
    assert r.status_code in (400, 422)


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_query_latency_not_regressed(base_url, squad_sample):
    times = []
    for row in squad_sample[:3]:
        t = time.perf_counter()
        httpx.post(f"{base_url}/query", json={"question": row["question"]}, timeout=30)
        times.append((time.perf_counter() - t) * 1000)
    p95 = sorted(times)[int(len(times) * 0.95)]
    stored = _baseline("query_latency", {"p95_ms": round(p95, 1)})
    assert p95 < stored["p95_ms"] * 1.5


# ── A/B ───────────────────────────────────────────────────────────────────────

def test_short_vs_long_question_latency(base_url):
    short = "What is RAG?"
    long_q = (
        "Can you provide a comprehensive explanation of retrieval-augmented generation, "
        "how it differs from standard language model inference, what databases it typically "
        "uses, and what are the main tradeoffs in its design compared to fine-tuning?"
    )

    def call(q):
        httpx.post(f"{base_url}/query", json={"question": q}, timeout=60)

    t_short = time.perf_counter()
    for _ in range(3):
        call(short)
    short_ms = (time.perf_counter() - t_short) / 3 * 1000

    t_long = time.perf_counter()
    for _ in range(3):
        call(long_q)
    long_ms = (time.perf_counter() - t_long) / 3 * 1000

    print(f"\n  short question avg={short_ms:.0f}ms  long question avg={long_ms:.0f}ms")
    assert short_ms < 30_000 and long_ms < 30_000
