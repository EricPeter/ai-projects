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


def test_translate_english_to_luganda(base_url, flores_sample):
    for row in flores_sample[:2]:
        r = httpx.post(
            f"{base_url}/translate",
            json={"text": row["text"], "source_language": "eng", "target_language": "lug"},
            timeout=30,
        )
        assert r.status_code == 200
        assert "translation" in r.json()
        assert len(r.json()["translation"]) > 0


def test_detect_and_translate(base_url):
    r = httpx.post(
        f"{base_url}/detect-and-translate",
        json={"text": "Good morning, how are you?"},
        timeout=30,
    )
    assert r.status_code == 200
    body = r.json()
    assert "detected_language" in body
    assert "translation" in body


# ── Regression ────────────────────────────────────────────────────────────────

def test_translation_response_schema_stable(base_url):
    r = httpx.post(
        f"{base_url}/translate",
        json={"text": "Hello", "source_language": "eng", "target_language": "lug"},
        timeout=30,
    )
    keys = sorted(r.json().keys())
    stored = _baseline("translate_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_latency_not_regressed(base_url):
    times = []
    for _ in range(3):
        t = time.perf_counter()
        httpx.post(
            f"{base_url}/translate",
            json={"text": "Hello", "source_language": "eng", "target_language": "lug"},
            timeout=30,
        )
        times.append((time.perf_counter() - t) * 1000)
    p95 = sorted(times)[-1]
    stored = _baseline("latency", {"p95_ms": round(p95, 1)})
    assert p95 < stored["p95_ms"] * 1.5


# ── A/B ── short vs long sentences ───────────────────────────────────────────

def test_short_vs_long_sentence_latency(base_url, flores_sample):
    short = "Hello."
    long_s = " ".join([row["text"] for row in flores_sample[:3]])[:400]

    t0 = time.perf_counter()
    for _ in range(3):
        httpx.post(f"{base_url}/translate", json={"text": short, "source_language": "eng", "target_language": "lug"}, timeout=30)
    short_avg = (time.perf_counter() - t0) / 3 * 1000

    t0 = time.perf_counter()
    for _ in range(3):
        httpx.post(f"{base_url}/translate", json={"text": long_s, "source_language": "eng", "target_language": "lug"}, timeout=30)
    long_avg = (time.perf_counter() - t0) / 3 * 1000

    print(f"\n  short sentence avg: {short_avg:.0f}ms  long sentence avg: {long_avg:.0f}ms")
    assert short_avg < 30_000 and long_avg < 30_000
