from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .monitor import compute_accuracy, compute_latency_stats


@dataclass
class Thresholds:
    min_accuracy: float = 0.85       # trigger if accuracy drops below this
    max_latency_p95_ms: float = 500  # trigger if p95 latency exceeds this
    min_labeled_samples: int = 50    # require at least this many labeled samples


def should_retrain(
    thresholds: Optional[Thresholds] = None,
    hours: float = 24.0,
) -> dict:
    """
    Evaluate current model health and return a decision with reasons.

    """
    t = thresholds or Thresholds()
    accuracy = compute_accuracy(hours)
    latency = compute_latency_stats(hours)
    reasons = []

    if accuracy["n"] < t.min_labeled_samples:
        return {
            "retrain": False,
            "reasons": [f"Only {accuracy['n']} labeled samples — need {t.min_labeled_samples}"],
            "accuracy": accuracy,
            "latency": latency,
        }

    if accuracy["accuracy"] is not None and accuracy["accuracy"] < t.min_accuracy:
        reasons.append(
            f"Accuracy {accuracy['accuracy']:.2%} below threshold {t.min_accuracy:.2%}"
        )

    if latency["p95_ms"] is not None and latency["p95_ms"] > t.max_latency_p95_ms:
        reasons.append(
            f"p95 latency {latency['p95_ms']:.0f}ms exceeds threshold {t.max_latency_p95_ms:.0f}ms"
        )

    return {
        "retrain": len(reasons) > 0,
        "reasons": reasons,
        "accuracy": accuracy,
        "latency": latency,
    }
