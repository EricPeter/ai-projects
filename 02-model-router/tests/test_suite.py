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


def test_chat_returns_response_and_model(base_url, squad_sample):
    for q in squad_sample[:2]:
        r = httpx.post(f"{base_url}/chat", json={"message": q}, timeout=30)
        assert r.status_code == 200
        body = r.json()
        assert "response" in body
        assert "model" in body


def test_costs_endpoint(base_url):
    r = httpx.get(f"{base_url}/costs", timeout=10)
    assert r.status_code == 200


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_chat_latency_not_regressed(base_url):
    times = []
    for msg in ["What is 2+2?", "Hello", "Translate 'cat' to French"]:
        t = time.perf_counter()
        httpx.post(f"{base_url}/chat", json={"message": msg}, timeout=30)
        times.append((time.perf_counter() - t) * 1000)
    p95 = sorted(times)[-1]
    stored = _baseline("chat_latency", {"p95_ms": round(p95, 1)})
    assert p95 < stored["p95_ms"] * 1.5


# ── A/B ───────────────────────────────────────────────────────────────────────

def test_cheap_vs_complex_query_routing(base_url):
    cheap_msg = "What is 2 + 2?"
    complex_msg = (
        "Write a complete red-black tree implementation in Python with insert, "
        "delete, and search, including all rotation operations."
    )

    t0 = time.perf_counter()
    r_cheap = httpx.post(f"{base_url}/chat", json={"message": cheap_msg}, timeout=60)
    cheap_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r_complex = httpx.post(f"{base_url}/chat", json={"message": complex_msg}, timeout=60)
    complex_ms = (time.perf_counter() - t0) * 1000

    cheap_model = r_cheap.json().get("model") or r_cheap.json().get("router_decision") or "N/A"
    complex_model = r_complex.json().get("model") or r_complex.json().get("router_decision") or "N/A"

    print(f"\n  cheap query: model={cheap_model}  {cheap_ms:.0f}ms")
    print(f"  complex query: model={complex_model}  {complex_ms:.0f}ms")

    assert r_cheap.status_code == 200 and r_complex.status_code == 200
