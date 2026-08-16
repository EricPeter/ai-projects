import pytest
import json
import numpy as np
from src.detector import _severity, compute_stats, DRIFT_THRESHOLDS


class TestSeverityClassification:
    def test_below_warning_returns_none(self):
        assert _severity("semantic_similarity", 0.01) is None

    def test_at_warning_threshold(self):
        assert _severity("semantic_similarity", 0.08) == "warning"

    def test_above_critical_threshold(self):
        assert _severity("semantic_similarity", 0.20) == "critical"

    def test_latency_warning(self):
        assert _severity("latency_p95", 0.35) == "warning"

    def test_latency_critical(self):
        assert _severity("latency_p95", 0.65) == "critical"

    def test_unknown_metric_returns_none(self):
        assert _severity("nonexistent_metric", 999.0) is None


class TestComputeStats:
    def _make_snapshots(self, latencies: list[float], tokens: list[int]) -> list[dict]:
        return [
            {"latency_ms": l, "tokens": t, "response": "", "embedding": json.dumps([0.0])}
            for l, t in zip(latencies, tokens)
        ]

    def test_empty_snapshots_returns_empty(self):
        assert compute_stats([]) == {}

    def test_p50_is_median(self):
        snaps = self._make_snapshots([100, 200, 300, 400, 500], [10] * 5)
        stats = compute_stats(snaps)
        assert stats["latency_p50_ms"] == 300.0

    def test_p95_greater_than_p50(self):
        latencies = list(range(1, 101))
        snaps = self._make_snapshots(latencies, [50] * 100)
        stats = compute_stats(snaps)
        assert stats["latency_p95_ms"] > stats["latency_p50_ms"]

    def test_avg_tokens_correct(self):
        snaps = self._make_snapshots([100.0] * 4, [10, 20, 30, 40])
        stats = compute_stats(snaps)
        assert stats["avg_tokens"] == 25.0

    def test_count_matches_input(self):
        snaps = self._make_snapshots([100.0] * 7, [50] * 7)
        assert compute_stats(snaps)["count"] == 7
