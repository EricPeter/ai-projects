from __future__ import annotations

import threading
from collections import deque

from prometheus_client import Counter, Gauge, Histogram

from .stats import compute_stats, welch_t_test, two_proportion_z_test

# ── Prometheus ────────────────────────────────────────────────────────────────
ab_requests_total = Counter(
    "ab_requests_total",
    "Total requests routed by the A/B router",
    labelnames=["variant", "status_code"],
)
ab_latency_seconds = Histogram(
    "ab_latency_seconds",
    "Upstream model latency",
    labelnames=["variant"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
ab_traffic_split = Gauge(
    "ab_experiment_traffic_split",
    "Current fraction of traffic routed to model B",
)
ab_samples_total = Gauge(
    "ab_experiment_samples_total",
    "Number of samples collected per variant",
    labelnames=["variant"],
)
ab_significance_achieved = Gauge(
    "ab_significance_achieved",
    "1 if the experiment has reached statistical significance, else 0",
)

_MAX_SAMPLES = 10_000


class ExperimentStore:
    """Thread-safe store for per-variant metrics and experiment state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latencies: dict[str, deque] = {
            "a": deque(maxlen=_MAX_SAMPLES),
            "b": deque(maxlen=_MAX_SAMPLES),
        }
        self._successes: dict[str, int] = {"a": 0, "b": 0}
        self._totals: dict[str, int] = {"a": 0, "b": 0}

    def record(self, variant: str, latency_s: float, success: bool) -> None:
        with self._lock:
            self._latencies[variant].append(latency_s * 1000)
            self._totals[variant] += 1
            if success:
                self._successes[variant] += 1
        ab_latency_seconds.labels(variant=variant).observe(latency_s)
        ab_samples_total.labels(variant=variant).set(self._totals[variant])

    def reset(self) -> None:
        with self._lock:
            for v in ("a", "b"):
                self._latencies[v].clear()
                self._successes[v] = 0
                self._totals[v] = 0
        ab_significance_achieved.set(0)

    def get_summary(self, min_samples: int = 100, alpha: float = 0.05) -> dict:
        with self._lock:
            a_lat = list(self._latencies["a"])
            b_lat = list(self._latencies["b"])
            a_suc, a_tot = self._successes["a"], self._totals["a"]
            b_suc, b_tot = self._successes["b"], self._totals["b"]

        ready = a_tot >= min_samples and b_tot >= min_samples
        lat_test = welch_t_test(a_lat, b_lat) if ready else {"significant": False, "reason": "insufficient samples"}
        rate_test = (
            two_proportion_z_test(a_suc, a_tot, b_suc, b_tot)
            if ready
            else {"significant": False, "reason": "insufficient samples"}
        )
        significant = ready and (lat_test.get("significant") or rate_test.get("significant"))
        ab_significance_achieved.set(1.0 if significant else 0.0)

        return {
            "variants": {
                "a": {**compute_stats(a_lat), "successes": a_suc, "total": a_tot},
                "b": {**compute_stats(b_lat), "successes": b_suc, "total": b_tot},
            },
            "latency_test": lat_test,
            "success_rate_test": rate_test,
            "significant": significant,
            "recommendation": (
                f"Promote model {lat_test.get('winner', '?')}"
                if significant
                else "Continue collecting data"
            ),
        }


store = ExperimentStore()
