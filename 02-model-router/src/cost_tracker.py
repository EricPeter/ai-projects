import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data/costs.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            complexity TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def record(request_id: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float, complexity: str):
    conn = _conn()
    conn.execute(
        "INSERT INTO requests (request_id, model, input_tokens, output_tokens, cost_usd, complexity) VALUES (?,?,?,?,?,?)",
        (request_id, model, input_tokens, output_tokens, cost_usd, complexity),
    )
    conn.commit()
    conn.close()


def summary() -> dict:
    conn = _conn()
    rows = conn.execute("""
        SELECT model, COUNT(*) as requests, SUM(cost_usd) as total_cost,
               SUM(input_tokens) as total_input, SUM(output_tokens) as total_output
        FROM requests
        GROUP BY model
    """).fetchall()
    conn.close()
    return [{"model": r[0], "requests": r[1], "total_cost_usd": round(r[2] or 0, 6),
             "total_input_tokens": r[3], "total_output_tokens": r[4]} for r in rows]
