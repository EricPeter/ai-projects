import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path("security_audit.jsonl")


def log_blocked(reason: str, client_ip: str, sanitized_input: str) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "blocked",
        "reason": reason,
        "client_ip": client_ip,
        "input_preview": sanitized_input[:100],
    }
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
