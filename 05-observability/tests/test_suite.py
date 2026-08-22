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


def test_chat_returns_200(base_url, chat_inputs):
    for msg in chat_inputs[:2]:
        r = httpx.post(f"{base_url}/chat", json={"message": msg}, timeout=30)
        assert r.status_code == 200


def test_metrics_endpoint_prometheus_format(base_url):
    r = httpx.get(f"{base_url}/metrics", timeout=10)
    assert r.status_code == 200
    assert "# HELP" in r.text or "# TYPE" in r.text


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_latency_not_regressed(base_url):
    times = []
    for _ in range(5):
        t = time.perf_counter()
        httpx.get(f"{base_url}/health", timeout=10)
        times.append((time.perf_counter() - t) * 1000)
    p95 = sorted(times)[-1]
    stored = _baseline("health_latency", {"p95_ms": round(p95, 1)})
    assert p95 < stored["p95_ms"] * 1.5


# ── A/B ───────────────────────────────────────────────────────────────────────

def test_single_vs_concurrent_requests(base_url):
    import threading

    msg = "Explain quantum entanglement in one sentence."

    # Sequential
    t0 = time.perf_counter()
    for _ in range(3):
        httpx.post(f"{base_url}/chat", json={"message": msg}, timeout=30)
    sequential_ms = (time.perf_counter() - t0) * 1000

    # Concurrent
    results = []
    def fire():
        t = time.perf_counter()
        httpx.post(f"{base_url}/chat", json={"message": msg}, timeout=30)
        results.append((time.perf_counter() - t) * 1000)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=fire) for _ in range(3)]
    for th in threads: th.start()
    for th in threads: th.join()
    concurrent_wall_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  sequential 3x: {sequential_ms:.0f}ms  concurrent 3x wall: {concurrent_wall_ms:.0f}ms")
    assert len(results) == 3
