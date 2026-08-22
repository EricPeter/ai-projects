"""
A/B test: verify model-b outperforms model-a in latency after sufficient traffic.
This is the core claim of the project — test it rigorously.
"""
import time

import httpx
import pytest


def _send_traffic(base_url: str, n: int = 300):
    httpx.post(f"{base_url}/experiment/reset", timeout=10)
    for i in range(n):
        httpx.post(f"{base_url}/infer", json={"text": f"sample request {i}"}, timeout=30)


def test_model_b_lower_latency_after_traffic(base_url):
    _send_traffic(base_url, n=300)

    body = httpx.get(f"{base_url}/experiment", timeout=10).json()
    a = body["variants"]["a"]
    b = body["variants"]["b"]

    assert a["n"] > 0 and b["n"] > 0, "both variants must have received traffic"

    print(f"\n  model-a  p50={a['mean']:.1f}ms  n={a['n']}")
    print(f"  model-b  p50={b['mean']:.1f}ms  n={b['n']}")

    # model-b stub has 80ms mean vs model-a's 100ms — should be faster
    assert b["mean"] < a["mean"], (
        f"model-b ({b['mean']:.1f}ms) should be faster than model-a ({a['mean']:.1f}ms)"
    )


def test_significance_reached_after_traffic(base_url):
    _send_traffic(base_url, n=300)
    body = httpx.get(f"{base_url}/experiment", timeout=10).json()
    assert body["significant"], "experiment should reach significance after 300 requests"
    assert body["recommendation"] in ("Promote model a", "Promote model b")


def test_dynamic_split_shifts_traffic(base_url):
    httpx.post(f"{base_url}/experiment/reset", timeout=10)

    # Start at 10%
    httpx.put(f"{base_url}/experiment/split", json={"split": 0.1}, timeout=10)
    n = 60
    for i in range(n):
        httpx.post(f"{base_url}/infer", json={"text": f"t{i}"}, timeout=30)
    body_10 = httpx.get(f"{base_url}/experiment", timeout=10).json()
    b_frac_10 = body_10["variants"]["b"]["n"] / n

    # Shift to 50%
    httpx.put(f"{base_url}/experiment/split", json={"split": 0.5}, timeout=10)
    httpx.post(f"{base_url}/experiment/reset", timeout=10)
    for i in range(n):
        httpx.post(f"{base_url}/infer", json={"text": f"t{i}"}, timeout=30)
    body_50 = httpx.get(f"{base_url}/experiment", timeout=10).json()
    b_frac_50 = body_50["variants"]["b"]["n"] / n

    print(f"\n  split=0.10 → b got {b_frac_10:.0%}")
    print(f"  split=0.50 → b got {b_frac_50:.0%}")
    assert b_frac_50 > b_frac_10, "increasing split should send more traffic to b"

    httpx.put(f"{base_url}/experiment/split", json={"split": 0.1}, timeout=10)
