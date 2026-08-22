from __future__ import annotations

import time

from .collector import get_labeled_window, labeled_count
from .monitor import health_report
from .trigger import Thresholds, should_retrain
from .trainer import train


def run_once(thresholds: Thresholds | None = None) -> dict:
    decision = should_retrain(thresholds)
    report = {
        "checked_at": time.time(),
        "health": health_report(),
        "decision": decision,
        "action": None,
    }

    if decision["retrain"]:
        rows = get_labeled_window()
        result = train(rows)
        report["action"] = result
        print(f"[pipeline] Retrained on {len(rows)} samples — reasons: {decision['reasons']}")
    else:
        print(f"[pipeline] No retraining needed — accuracy OK, {labeled_count()} labeled samples")

    return report


def run_loop(interval_s: float = 3600.0, thresholds: Thresholds | None = None) -> None:
    print(f"[pipeline] Starting — checking every {interval_s}s")
    while True:
        run_once(thresholds)
        time.sleep(interval_s)
