import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path("outputs/audit_trail.jsonl")


def log(agent: str, action: str, input_summary: str, output_summary: str) -> None:
    AUDIT_FILE.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "action": action,
        "input": input_summary[:500],
        "output": output_summary[:500],
    }
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
