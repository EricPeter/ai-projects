import argparse
import random
import sys
import time
sys.path.insert(0, ".")

from src.collector import log_inference, log_label
from src.pipeline import run_once
from src.trigger import Thresholds


def run(n_clean: int, n_drift: int, interval: float) -> None:
    from datasets import load_dataset

    ds = load_dataset("ag_news", split="test")
    labels = ["World", "Sports", "Business", "Sci/Tech"]

    print(f"Phase 1 — {n_clean} clean samples (model performs well)")
    for i in range(n_clean):
        row = ds[i % len(ds)]
        true_label = labels[row["label"]]
        pred = true_label if random.random() > 0.05 else random.choice(labels)
        rid = log_inference({"text": row["text"][:100]}, pred, latency_ms=random.uniform(30, 80))
        log_label(rid, true_label)
        if (i + 1) % 20 == 0:
            print(f"  logged {i+1}/{n_clean}")
        time.sleep(interval)

    print(f"\nPhase 2 — {n_drift} drift samples (model degrades)")
    for i in range(n_drift):
        row = ds[(n_clean + i) % len(ds)]
        true_label = labels[row["label"]]
        pred = random.choice(labels)  # random predictions — accuracy ~25%
        rid = log_inference({"text": row["text"][:100]}, pred, latency_ms=random.uniform(200, 600))
        log_label(rid, true_label)
        if (i + 1) % 20 == 0:
            print(f"  logged {i+1}/{n_drift}")
        time.sleep(interval)

    print("\nRunning pipeline check...")
    t = Thresholds(min_accuracy=0.70, max_latency_p95_ms=300, min_labeled_samples=30)
    report = run_once(t)
    print(f"Decision: retrain={report['decision']['retrain']}")
    print(f"Reasons:  {report['decision']['reasons']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=int, default=60)
    parser.add_argument("--drift", type=int, default=60)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()
    run(args.clean, args.drift, args.interval)
