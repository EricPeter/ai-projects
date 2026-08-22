"""
Shared test helpers imported by every project's conftest.py.

Usage:
    from tests.shared import assert_latency_p95, save_or_compare_baseline, ab_latency_compare
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import httpx


def get(base_url: str, path: str, **kw) -> httpx.Response:
    return httpx.get(f"{base_url}{path}", timeout=30, **kw)


def post(base_url: str, path: str, payload: dict, **kw) -> httpx.Response:
    return httpx.post(f"{base_url}{path}", json=payload, timeout=60, **kw)


def measure_latency(fn, n: int = 10) -> list[float]:
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return times


def p50(vals: list[float]) -> float:
    return statistics.median(vals)


def p95(vals: list[float]) -> float:
    s = sorted(vals)
    return s[max(0, int(len(s) * 0.95) - 1)]


def save_or_compare_baseline(path: Path, current: dict, tol: float = 0.20) -> str:
    """
    Returns "created", "match", or "drift:<field>=<old>-><new>" string.
    Compares numeric values within tol (20% by default).
    Non-numeric fields are compared for equality.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(current, indent=2))
        return "created"

    baseline = json.loads(path.read_text())
    drifted = []
    for k, v in baseline.items():
        c = current.get(k)
        if isinstance(v, (int, float)) and isinstance(c, (int, float)):
            if v != 0 and abs(c - v) / abs(v) > tol:
                drifted.append(f"{k}={v:.3g}->{c:.3g}")
        elif c != v:
            drifted.append(f"{k}={v!r}->{c!r}")
    return "drift:" + ", ".join(drifted) if drifted else "match"


def ab_latency_compare(fn_a, fn_b, n: int = 10) -> dict:
    """
    Run fn_a and fn_b n times each and return a comparison dict.
    """
    a = measure_latency(fn_a, n)
    b = measure_latency(fn_b, n)
    return {
        "a_p50_ms": round(p50(a), 1),
        "b_p50_ms": round(p50(b), 1),
        "a_p95_ms": round(p95(a), 1),
        "b_p95_ms": round(p95(b), 1),
        "delta_pct": round((p50(b) - p50(a)) / max(p50(a), 0.001) * 100, 1),
    }
