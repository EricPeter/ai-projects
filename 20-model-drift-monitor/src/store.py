import aiosqlite
import json
from datetime import datetime

DB_PATH = "data/drift.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                probe_id TEXT NOT NULL,
                response TEXT NOT NULL,
                embedding TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                probe_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                baseline_value REAL NOT NULL,
                current_value REAL NOT NULL,
                delta REAL NOT NULL,
                severity TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await db.commit()


async def save_snapshot(model: str, probe_id: str, response: str, embedding: list[float], latency_ms: float, tokens: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO snapshots (model, probe_id, response, embedding, latency_ms, tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model, probe_id, response, json.dumps(embedding), latency_ms, tokens, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_snapshots(model: str, probe_id: str, limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM snapshots WHERE model=? AND probe_id=? ORDER BY created_at DESC LIMIT ?",
            (model, probe_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def save_alert(model: str, probe_id: str, metric: str, baseline: float, current: float, severity: str):
    delta = current - baseline
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO drift_alerts (model, probe_id, metric, baseline_value, current_value, delta, severity, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model, probe_id, metric, baseline, current, delta, severity, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_alerts(model: str | None = None, limit: int = 20) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if model:
            async with db.execute(
                "SELECT * FROM drift_alerts WHERE model=? ORDER BY created_at DESC LIMIT ?",
                (model, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM drift_alerts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
    return [dict(r) for r in rows]
