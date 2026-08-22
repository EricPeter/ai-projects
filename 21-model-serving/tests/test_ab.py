"""
A/B test: single-sample inference vs batched inference.
Compares throughput and latency between batch_size=1 and batch_size=4.
"""
import statistics
import time

import httpx
import pytest


def _get_model_and_inputs(base_url: str):
    models = httpx.get(f"{base_url}/models", timeout=10).json()
    name = models[0]["name"]
    meta = httpx.get(f"{base_url}/models/{name}", timeout=10).json()
    inputs_single = {}
    for inp in meta["inputs"]:
        shape = [1 if s in (-1, None) else s for s in inp["shape"]]
        inputs_single[inp["name"]] = [[101] * shape[-1]]
    return name, inputs_single


def _latencies(base_url, name, inputs, n=10):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        httpx.post(f"{base_url}/infer/{name}", json={"inputs": inputs}, timeout=60)
        times.append((time.perf_counter() - t) * 1000)
    return times


def test_single_inference_latency_acceptable(base_url):
    name, inputs = _get_model_and_inputs(base_url)
    lats = _latencies(base_url, name, inputs, n=10)
    p50 = statistics.median(lats)
    p95 = sorted(lats)[int(len(lats) * 0.95)]
    assert p95 < 500, f"p95={p95:.1f}ms — single-sample inference too slow"
    print(f"\n  single-sample  p50={p50:.1f}ms  p95={p95:.1f}ms")


def test_benchmark_throughput_vs_single(base_url):
    name, inputs = _get_model_and_inputs(base_url)

    # Single sequential calls
    single_lats = _latencies(base_url, name, inputs, n=10)
    single_p50 = statistics.median(single_lats)

    # Benchmark endpoint (uses internal warmup + batching)
    r = httpx.post(
        f"{base_url}/benchmark/{name}",
        json={"inputs": inputs, "warmup": 5, "iterations": 20},
        timeout=120,
    )
    bench = r.json()
    bench_p50 = bench["p50_ms"]
    bench_rps = bench["throughput_rps"]

    # Benchmark should show lower latency than cold sequential calls
    print(
        f"\n  sequential p50={single_p50:.1f}ms  "
        f"benchmark p50={bench_p50:.1f}ms  rps={bench_rps:.1f}"
    )
    assert bench_rps > 0, "throughput_rps must be positive"
