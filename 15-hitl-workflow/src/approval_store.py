import asyncio
from datetime import datetime
from pathlib import Path
import sqlite3

DB_PATH = Path("data/approvals.db")


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approvals (
            workflow_id TEXT PRIMARY KEY,
            pending_action TEXT,
            requested_at TEXT,
            decision TEXT,
            decided_by TEXT,
            decided_at TEXT,
            feedback TEXT
        )
    """)
    conn.commit()
    return conn


def create(workflow_id: str, pending_action: str):
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO approvals (workflow_id, pending_action, requested_at) VALUES (?,?,?)",
        (workflow_id, pending_action, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def decide(workflow_id: str, decision: str, decided_by: str, feedback: str = ""):
    conn = _conn()
    conn.execute(
        "UPDATE approvals SET decision=?, decided_by=?, decided_at=?, feedback=? WHERE workflow_id=?",
        (decision, decided_by, datetime.utcnow().isoformat(), feedback, workflow_id),
    )
    conn.commit()
    conn.close()


def get(workflow_id: str) -> dict | None:
    conn = _conn()
    row = conn.execute("SELECT * FROM approvals WHERE workflow_id=?", (workflow_id,)).fetchone()
    conn.close()
    if not row:
        return None
    cols = ["workflow_id", "pending_action", "requested_at", "decision", "decided_by", "decided_at", "feedback"]
    return dict(zip(cols, row))


def list_pending() -> list[dict]:
    conn = _conn()
    rows = conn.execute("SELECT * FROM approvals WHERE decision IS NULL").fetchall()
    conn.close()
    cols = ["workflow_id", "pending_action", "requested_at", "decision", "decided_by", "decided_at", "feedback"]
    return [dict(zip(cols, r)) for r in rows]
