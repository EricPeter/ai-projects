import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_FILE = Path("eval_history.jsonl")


def save_run(scores: dict[str, float], metadata: dict[str, Any] | None = None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        **(metadata or {}),
    }
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    return [json.loads(line) for line in HISTORY_FILE.read_text().splitlines() if line.strip()]


def trend(metric: str) -> str:
    history = load_history()
    if len(history) < 2:
        return "insufficient_data"
    last = history[-1]["scores"].get(metric, 0)
    prev = history[-2]["scores"].get(metric, 0)
    if last > prev + 0.01:
        return "improving"
    if last < prev - 0.01:
        return "degrading"
    return "stable"
