from __future__ import annotations

from .collector import get_labeled_window


def compute_accuracy(hours: float = 24.0) -> dict:
    """Accuracy of the current model over the last N hours of labeled data."""
    rows = get_labeled_window(hours)
    if not rows:
        return {"accuracy": None, "n": 0, "hours": hours}
    correct = sum(1 for r in rows if r["prediction"] == r["label"])
    return {
        "accuracy": round(correct / len(rows), 4),
        "correct": correct,
        "n": len(rows),
        "hours": hours,
    }


def compute_latency_stats(hours: float = 24.0) -> dict:
    """P50/P95 latency from the collector database."""
    import sqlite3, numpy as np
    from pathlib import Path
    from .collector import DB_PATH
    import time

    if not DB_PATH.exists():
        return {"p50_ms": None, "p95_ms": None, "n": 0}

    cutoff = time.time() - hours * 3600
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT latency_ms FROM inferences WHERE ts >= ? AND latency_ms IS NOT NULL",
        (cutoff,),
    ).fetchall()
    conn.close()

    if not rows:
        return {"p50_ms": None, "p95_ms": None, "n": 0}

    arr = np.array([r[0] for r in rows])
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "n": len(arr),
    }


def health_report(hours: float = 24.0) -> dict:
    return {
        "accuracy": compute_accuracy(hours),
        "latency": compute_latency_stats(hours),
    }
