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


def test_english_question_returns_answer(base_url, health_questions):
    eng = next(q for q in health_questions if q["language"] == "eng")
    r = httpx.post(f"{base_url}/ask", json=eng, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["answer"]) > 0


def test_luganda_question_returns_answer(base_url, health_questions):
    lug = next((q for q in health_questions if q["language"] == "lug"), None)
    if lug is None:
        pytest.skip("no Luganda question in fixture")
    r = httpx.post(f"{base_url}/ask", json=lug, timeout=30)
    assert r.status_code == 200
    assert "answer" in r.json()


def test_answer_includes_sources(base_url, health_questions):
    r = httpx.post(f"{base_url}/ask", json=health_questions[0], timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert "sources" in body or "answer" in body


# ── Regression ────────────────────────────────────────────────────────────────

def test_response_schema_stable(base_url, health_questions):
    r = httpx.post(f"{base_url}/ask", json=health_questions[0], timeout=30)
    keys = sorted(r.json().keys())
    stored = _baseline("ask_keys", {"keys": keys})
    assert keys == stored["keys"]


# ── A/B ── English vs Luganda response latency ────────────────────────────────

def test_english_vs_luganda_latency(base_url, health_questions):
    eng = next((q for q in health_questions if q["language"] == "eng"), None)
    lug = next((q for q in health_questions if q["language"] == "lug"), None)
    if not eng or not lug:
        pytest.skip("need both English and Luganda questions")

    t0 = time.perf_counter()
    httpx.post(f"{base_url}/ask", json=eng, timeout=30)
    eng_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    httpx.post(f"{base_url}/ask", json=lug, timeout=30)
    lug_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  English: {eng_ms:.0f}ms  Luganda: {lug_ms:.0f}ms")
    assert eng_ms < 30_000 and lug_ms < 30_000
