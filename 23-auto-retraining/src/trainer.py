from __future__ import annotations

import json
import time
from pathlib import Path

MODEL_PATH = Path("data/model.json")


def train(rows: list[dict]) -> dict:
    from collections import Counter

    if not rows:
        return {"error": "no training data"}

    label_counts = Counter(r["label"] for r in rows)
    majority = label_counts.most_common(1)[0][0]

    model = {
        "type": "majority_class",
        "majority": majority,
        "label_counts": dict(label_counts),
        "trained_on": len(rows),
        "trained_at": time.time(),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2))

    return {"status": "trained", "model": model}


def predict(features: dict) -> str:
    if not MODEL_PATH.exists():
        return "unknown"
    model = json.loads(MODEL_PATH.read_text())
    return model.get("majority", "unknown")
