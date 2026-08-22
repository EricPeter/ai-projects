import json
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"


def _baseline(name, current):
    BASELINES.mkdir(exist_ok=True)
    path = BASELINES / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline for '{name}' created")
    return json.loads(path.read_text())


def test_experiment_schema_stable(base_url):
    body = httpx.get(f"{base_url}/experiment", timeout=10).json()
    keys = sorted(body.keys())
    stored = _baseline("experiment_keys", {"keys": keys})
    assert keys == stored["keys"], "experiment response schema changed"


def test_split_respected_within_tolerance(base_url):
    httpx.post(f"{base_url}/experiment/reset", timeout=10)
    httpx.put(f"{base_url}/experiment/split", json={"split": 0.2}, timeout=10)

    b_count = 0
    n = 100
    for _ in range(n):
        r = httpx.post(f"{base_url}/infer", json={"text": "sample"}, timeout=30)
        if r.headers.get("x-model-variant") == "b":
            b_count += 1

    actual_split = b_count / n
    assert abs(actual_split - 0.2) < 0.12, (
        f"traffic split {actual_split:.2f} deviates >12% from configured 0.20"
    )

    httpx.put(f"{base_url}/experiment/split", json={"split": 0.1}, timeout=10)


def test_variant_stats_types(base_url):
    for _ in range(20):
        httpx.post(f"{base_url}/infer", json={"text": "test"}, timeout=30)

    body = httpx.get(f"{base_url}/experiment", timeout=10).json()
    for variant in ("a", "b"):
        stats = body["variants"].get(variant, {})
        for field in ("mean", "p95", "n"):
            assert field in stats, f"variants.{variant}.{field} missing"
