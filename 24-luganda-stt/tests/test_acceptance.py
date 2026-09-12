import json
from pathlib import Path

import httpx
import pytest


def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "boosted_words" in body
    assert "spell_checker" in body
    assert "ngram_lm" in body


def test_health_reports_model(base_url):
    body = httpx.get(f"{base_url}/health", timeout=10).json()
    assert "canary" in body["model"].lower() or body["model"] != ""


def test_word_boost_list_endpoint(base_url):
    r = httpx.get(f"{base_url}/word-boost/list", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "words" in body
    assert "count" in body
    assert body["count"] >= 0


def test_transcribe_endpoint_accepts_wav(base_url, sample_wav_bytes):
    r = httpx.post(
        f"{base_url}/transcribe",
        files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        timeout=60,
    )
    assert r.status_code == 200
    body = r.json()
    assert "transcript" in body
    assert "transcript_corrected" in body
    assert "translation" in body
    assert "latency_ms" in body
    assert body["latency_ms"] > 0


def test_transcribe_returns_mode(base_url, sample_wav_bytes):
    r = httpx.post(
        f"{base_url}/transcribe?mode=fast",
        files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
        timeout=60,
    )
    assert r.status_code == 200
    assert r.json()["mode"] in ("fast", "rich")


def test_add_word_boost(base_url):
    r = httpx.post(f"{base_url}/word-boost/add?word=omubulamu", timeout=10)
    assert r.status_code == 200
    assert r.json()["added"] == "omubulamu"
