import json
import time
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"


def _save_or_check(name: str, current: dict, tol: float = 0.25):
    BASELINES.mkdir(exist_ok=True)
    path = BASELINES / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline for '{name}' created — rerun to compare")
    stored = json.loads(path.read_text())
    return stored, current


def test_health_schema_unchanged(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    keys = sorted(r.json().keys())
    stored, current = _save_or_check("health_keys", {"keys": keys})
    assert current["keys"] == stored["keys"], "health response schema changed"


def test_inference_latency_not_regressed(base_url):
    models = httpx.get(f"{base_url}/models", timeout=10).json()
    name = models[0]["name"]
    meta = httpx.get(f"{base_url}/models/{name}", timeout=10).json()

    inputs = {}
    for inp in meta["inputs"]:
        shape = [1 if s in (-1, None) else s for s in inp["shape"]]
        inputs[inp["name"]] = [[101] * shape[-1]]

    latencies = []
    for _ in range(10):
        t = time.perf_counter()
        httpx.post(f"{base_url}/infer/{name}", json={"inputs": inputs}, timeout=60)
        latencies.append((time.perf_counter() - t) * 1000)

    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    stored, _ = _save_or_check("latency", {"p95_ms": round(p95, 1)})
    baseline_p95 = stored["p95_ms"]
    assert p95 < baseline_p95 * 1.5, (
        f"p95 latency regressed: {p95:.1f}ms vs baseline {baseline_p95:.1f}ms"
    )


def test_model_count_stable(base_url):
    count = len(httpx.get(f"{base_url}/models", timeout=10).json())
    stored, current = _save_or_check("model_count", {"count": count})
    assert current["count"] == stored["count"], (
        f"model count changed: {stored['count']} → {current['count']}"
    )
