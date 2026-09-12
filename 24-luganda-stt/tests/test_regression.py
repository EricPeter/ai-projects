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


def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_transcribe_latency_not_regressed(base_url, sample_wav_bytes):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        httpx.post(
            f"{base_url}/transcribe",
            files={"file": ("t.wav", sample_wav_bytes, "audio/wav")},
            timeout=120,
        )
        times.append((time.perf_counter() - t0) * 1000)
    p95 = sorted(times)[-1]
    stored = _baseline("transcribe_latency", {"p95_ms": round(p95, 1)})
    assert p95 < stored["p95_ms"] * 1.5, f"Latency regressed: {p95:.0f}ms > {stored['p95_ms'] * 1.5:.0f}ms"


def test_boosted_word_count_stable(base_url):
    count = httpx.get(f"{base_url}/word-boost/list", timeout=10).json()["count"]
    stored = _baseline("word_boost_count", {"count": count})
    # Word count should never drop (adding is ok, removing is a regression)
    assert count >= stored["count"], f"Boosted word count dropped: {count} < {stored['count']}"


def test_spell_checker_active(base_url):
    body = httpx.get(f"{base_url}/health", timeout=10).json()
    stored = _baseline("spell_checker_active", {"spell_checker": body["spell_checker"]})
    # Spell checker being active is the reference state; it going inactive is a regression
    if stored["spell_checker"]:
        assert body["spell_checker"], "Spell checker was active but is now inactive"
