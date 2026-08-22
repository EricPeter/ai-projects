import httpx
import pytest


def test_health_ok(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "labeled_last_24h" in body
    assert "retrain_needed" in body


def test_log_inference(base_url):
    r = httpx.post(
        f"{base_url}/infer",
        json={"features": {"text": "Stocks rise on positive earnings."}, "prediction": "Business", "latency_ms": 42.0},
        timeout=10,
    )
    assert r.status_code == 200
    assert "id" in r.json()
    assert r.json()["id"] > 0


def test_log_label_links_to_inference(base_url):
    infer_r = httpx.post(
        f"{base_url}/infer",
        json={"features": {"text": "Local team wins championship."}, "prediction": "Sports", "latency_ms": 38.0},
        timeout=10,
    )
    inf_id = infer_r.json()["id"]

    label_r = httpx.post(f"{base_url}/label/{inf_id}", json={"label": "Sports"}, timeout=10)
    assert label_r.status_code == 200
    assert label_r.json()["ok"] is True


def test_health_reflects_labeled_count(base_url, ag_news_sample):
    before = httpx.get(f"{base_url}/health", timeout=10).json()["labeled_last_24h"]

    for row in ag_news_sample[:5]:
        infer_r = httpx.post(
            f"{base_url}/infer",
            json={"features": {"text": row["text"][:300]}, "prediction": row["label"], "latency_ms": 30.0},
            timeout=10,
        )
        inf_id = infer_r.json()["id"]
        httpx.post(f"{base_url}/label/{inf_id}", json={"label": row["label"]}, timeout=10)

    after = httpx.get(f"{base_url}/health", timeout=10).json()["labeled_last_24h"]
    assert after >= before + 5, "labeled count did not increase after logging labeled inferences"
