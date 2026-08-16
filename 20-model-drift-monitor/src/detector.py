import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
import json
from .store import get_snapshots, save_alert
from .probes import PROBE_MAP

DRIFT_THRESHOLDS = {
    "semantic_similarity": {"warning": 0.08, "critical": 0.15},
    "keyword_hit_rate": {"warning": 0.10, "critical": 0.20},
    "latency_p95": {"warning": 0.30, "critical": 0.60},
}

BASELINE_WINDOW = 20
CURRENT_WINDOW = 5


async def detect_drift(model: str, probe_id: str) -> list[dict]:
    snapshots = await get_snapshots(model, probe_id, limit=BASELINE_WINDOW + CURRENT_WINDOW)
    if len(snapshots) < BASELINE_WINDOW + CURRENT_WINDOW:
        return []

    baseline = snapshots[CURRENT_WINDOW:]
    current = snapshots[:CURRENT_WINDOW]
    alerts = []

    semantic_alert = await _check_semantic_drift(model, probe_id, baseline, current)
    if semantic_alert:
        alerts.append(semantic_alert)

    kw_alert = await _check_keyword_drift(model, probe_id, baseline, current)
    if kw_alert:
        alerts.append(kw_alert)

    latency_alert = await _check_latency_drift(model, probe_id, baseline, current)
    if latency_alert:
        alerts.append(latency_alert)

    return alerts


async def _check_semantic_drift(model: str, probe_id: str, baseline: list[dict], current: list[dict]) -> dict | None:
    baseline_embs = np.array([json.loads(s["embedding"]) for s in baseline])
    current_embs = np.array([json.loads(s["embedding"]) for s in current])

    baseline_centroid = baseline_embs.mean(axis=0)
    current_centroid = current_embs.mean(axis=0)
    drift = float(cosine(baseline_centroid, current_centroid))

    severity = _severity("semantic_similarity", drift)
    if severity:
        baseline_val = float(np.mean([cosine(e, baseline_centroid) for e in baseline_embs]))
        await save_alert(model, probe_id, "semantic_similarity", baseline_val, drift, severity)
        return {"metric": "semantic_similarity", "baseline": baseline_val, "current": drift, "severity": severity}
    return None


async def _check_keyword_drift(model: str, probe_id: str, baseline: list[dict], current: list[dict]) -> dict | None:
    probe = PROBE_MAP.get(probe_id)
    if not probe or not probe.expected_keywords:
        return None

    def hit_rate(snaps: list[dict]) -> float:
        hits = sum(
            1 for s in snaps
            if any(kw.lower() in s["response"].lower() for kw in probe.expected_keywords)
        )
        return hits / len(snaps)

    baseline_rate = hit_rate(baseline)
    current_rate = hit_rate(current)
    delta = abs(baseline_rate - current_rate)

    severity = _severity("keyword_hit_rate", delta)
    if severity:
        await save_alert(model, probe_id, "keyword_hit_rate", baseline_rate, current_rate, severity)
        return {"metric": "keyword_hit_rate", "baseline": baseline_rate, "current": current_rate, "severity": severity}
    return None


async def _check_latency_drift(model: str, probe_id: str, baseline: list[dict], current: list[dict]) -> dict | None:
    baseline_p95 = float(np.percentile([s["latency_ms"] for s in baseline], 95))
    current_p95 = float(np.percentile([s["latency_ms"] for s in current], 95))

    if baseline_p95 == 0:
        return None

    relative_change = (current_p95 - baseline_p95) / baseline_p95
    severity = _severity("latency_p95", relative_change)
    if severity:
        await save_alert(model, probe_id, "latency_p95", baseline_p95, current_p95, severity)
        return {"metric": "latency_p95_ms", "baseline": baseline_p95, "current": current_p95, "severity": severity}
    return None


def _severity(metric: str, value: float) -> str | None:
    thresholds = DRIFT_THRESHOLDS.get(metric, {})
    if value >= thresholds.get("critical", float("inf")):
        return "critical"
    if value >= thresholds.get("warning", float("inf")):
        return "warning"
    return None


def compute_stats(snapshots: list[dict]) -> dict:
    if not snapshots:
        return {}
    latencies = [s["latency_ms"] for s in snapshots]
    tokens = [s["tokens"] for s in snapshots]
    return {
        "count": len(snapshots),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "avg_tokens": float(np.mean(tokens)),
    }
