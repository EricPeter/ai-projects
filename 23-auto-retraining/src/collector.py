from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/inference.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inferences (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            features    TEXT    NOT NULL,
            prediction  TEXT    NOT NULL,
            label       TEXT,
            latency_ms  REAL
        )
    """)
    conn.commit()
    return conn


def log_inference(features: dict, prediction: str, latency_ms: float) -> int:
    import json
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO inferences (ts, features, prediction, latency_ms) VALUES (?,?,?,?)",
            (time.time(), json.dumps(features), prediction, latency_ms),
        )
        return cur.lastrowid


def log_label(inference_id: int, label: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE inferences SET label=? WHERE id=?", (label, inference_id)
        )


def get_labeled_window(hours: float = 24.0) -> list[dict]:
    import json
    cutoff = time.time() - hours * 3600
    with _conn() as conn:
        rows = conn.execute(
            "SELECT features, prediction, label FROM inferences "
            "WHERE label IS NOT NULL AND ts >= ?",
            (cutoff,),
        ).fetchall()
    return [
        {"features": json.loads(r[0]), "prediction": r[1], "label": r[2]}
        for r in rows
    ]


def labeled_count(hours: float = 24.0) -> int:
    cutoff = time.time() - hours * 3600
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM inferences WHERE label IS NOT NULL AND ts >= ?",
            (cutoff,),
        ).fetchone()[0]
