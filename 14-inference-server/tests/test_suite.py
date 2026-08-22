import json
import time
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"
MODEL = "facebook/opt-125m"


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


def test_openai_compatible_chat_completions(base_url, squad_questions):
    for q in squad_questions[:2]:
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": q}]},
            timeout=60,
        )
        assert r.status_code == 200
        body = r.json()
        assert "choices" in body
        assert len(body["choices"]) > 0
        assert "message" in body["choices"][0]


def test_metrics_prometheus_format(base_url):
    r = httpx.get(f"{base_url}/metrics", timeout=10)
    assert r.status_code == 200
    assert "# HELP" in r.text or "# TYPE" in r.text


# ── Regression ────────────────────────────────────────────────────────────────

def test_response_schema_stable(base_url):
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]},
        timeout=60,
    )
    keys = sorted(r.json().keys())
    stored = _baseline("completion_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_latency_not_regressed(base_url):
    times = []
    for _ in range(3):
        t = time.perf_counter()
        httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]},
            timeout=60,
        )
        times.append((time.perf_counter() - t) * 1000)
    p95 = sorted(times)[-1]
    stored = _baseline("latency", {"p95_ms": round(p95, 1)})
    assert p95 < stored["p95_ms"] * 1.5


# ── A/B ── single-turn vs multi-turn ─────────────────────────────────────────

def test_single_vs_multiturn_latency(base_url):
    single = [{"role": "user", "content": "What is Python?"}]
    multi = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "What are its main features?"},
    ]

    t0 = time.perf_counter()
    httpx.post(f"{base_url}/v1/chat/completions", json={"model": MODEL, "messages": single}, timeout=60)
    single_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    httpx.post(f"{base_url}/v1/chat/completions", json={"model": MODEL, "messages": multi}, timeout=60)
    multi_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  single-turn: {single_ms:.0f}ms  multi-turn: {multi_ms:.0f}ms")
    assert single_ms < 120_000 and multi_ms < 120_000
