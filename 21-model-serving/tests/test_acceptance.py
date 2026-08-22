import httpx
import pytest


def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models" in body


def test_models_list_not_empty(base_url):
    r = httpx.get(f"{base_url}/models", timeout=10)
    assert r.status_code == 200
    models = r.json()
    assert isinstance(models, list)
    assert len(models) >= 1, "at least one model must be loaded"


def test_model_metadata(base_url):
    models = httpx.get(f"{base_url}/models", timeout=10).json()
    name = models[0]["name"]
    r = httpx.get(f"{base_url}/models/{name}", timeout=10)
    assert r.status_code == 200
    meta = r.json()
    assert "inputs" in meta and "outputs" in meta


def test_infer_returns_outputs_and_latency(base_url):
    models = httpx.get(f"{base_url}/models", timeout=10).json()
    name = models[0]["name"]
    meta = httpx.get(f"{base_url}/models/{name}", timeout=10).json()

    # Build minimal inputs matching the model's expected input shapes
    inputs = {}
    for inp in meta["inputs"]:
        shape = [1 if s in (-1, None) else s for s in inp["shape"]]
        inputs[inp["name"]] = [[101] * shape[-1]]

    r = httpx.post(f"{base_url}/infer/{name}", json={"inputs": inputs}, timeout=60)
    assert r.status_code == 200
    body = r.json()
    assert "outputs" in body
    assert "latency_ms" in body
    assert body["latency_ms"] > 0


def test_benchmark_returns_percentiles(base_url):
    models = httpx.get(f"{base_url}/models", timeout=10).json()
    name = models[0]["name"]
    meta = httpx.get(f"{base_url}/models/{name}", timeout=10).json()

    inputs = {}
    for inp in meta["inputs"]:
        shape = [1 if s in (-1, None) else s for s in inp["shape"]]
        inputs[inp["name"]] = [[101] * shape[-1]]

    r = httpx.post(
        f"{base_url}/benchmark/{name}",
        json={"inputs": inputs, "warmup": 5, "iterations": 20},
        timeout=120,
    )
    assert r.status_code == 200
    body = r.json()
    for field in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "throughput_rps"):
        assert field in body, f"missing field: {field}"
    assert body["p95_ms"] >= body["p50_ms"]


def test_inference_on_sst2_sample(base_url, sst2_sample):
    models = httpx.get(f"{base_url}/models", timeout=10).json()
    name = models[0]["name"]
    meta = httpx.get(f"{base_url}/models/{name}", timeout=10).json()

    for row in sst2_sample[:3]:
        inputs = {}
        for inp in meta["inputs"]:
            shape = [1 if s in (-1, None) else s for s in inp["shape"]]
            inputs[inp["name"]] = [[101] * shape[-1]]
        r = httpx.post(f"{base_url}/infer/{name}", json={"inputs": inputs}, timeout=60)
        assert r.status_code == 200, f"failed for: {row['sentence'][:50]}"
