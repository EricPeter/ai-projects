import httpx
import pytest


def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_infer_returns_variant_header(base_url):
    r = httpx.post(f"{base_url}/infer", json={"text": "hello"}, timeout=30)
    assert r.status_code == 200
    assert "x-model-variant" in r.headers
    assert r.headers["x-model-variant"] in ("a", "b")


def test_experiment_response_shape(base_url):
    r = httpx.get(f"{base_url}/experiment", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "variants" in body
    assert "a" in body["variants"] and "b" in body["variants"]
    assert "significant" in body
    assert "recommendation" in body


def test_split_update(base_url):
    r = httpx.put(f"{base_url}/experiment/split", json={"split": 0.3}, timeout=10)
    assert r.status_code == 200

    r = httpx.put(f"{base_url}/experiment/split", json={"split": 0.1}, timeout=10)
    assert r.status_code == 200


def test_experiment_reset(base_url):
    r = httpx.post(f"{base_url}/experiment/reset", timeout=10)
    assert r.status_code == 200


def test_infer_with_ag_news_texts(base_url, ag_news_sample):
    for text in ag_news_sample[:5]:
        r = httpx.post(f"{base_url}/infer", json={"text": text[:200]}, timeout=30)
        assert r.status_code == 200
        assert "x-model-variant" in r.headers
