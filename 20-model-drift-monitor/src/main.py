import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from .store import init_db, get_snapshots, get_alerts
from .runner import run_probe
from .detector import detect_drift, compute_stats
from .probes import PROBES, PROBE_MAP

DEFAULT_MODEL = os.getenv("MODEL", "claude-haiku-4-5-20251001")
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL_SECONDS", "300"))

_monitor_task: asyncio.Task | None = None


async def _monitor_loop():
    while True:
        for probe in PROBES:
            try:
                await run_probe(DEFAULT_MODEL, probe)
                await detect_drift(DEFAULT_MODEL, probe.id)
            except Exception:
                pass
        await asyncio.sleep(MONITOR_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    global _monitor_task
    _monitor_task = asyncio.create_task(_monitor_loop())
    yield
    _monitor_task.cancel()


app = FastAPI(title="Model Drift Monitor", lifespan=lifespan)


class ProbeRequest(BaseModel):
    model: str = DEFAULT_MODEL
    probe_id: str | None = None


@app.post("/probe")
async def run_single_probe(req: ProbeRequest):
    probes = [PROBE_MAP[req.probe_id]] if req.probe_id else PROBES
    results = []
    alerts = []
    for probe in probes:
        result = await run_probe(req.model, probe)
        drift = await detect_drift(req.model, probe.id)
        results.append(result)
        alerts.extend(drift)
    return {"results": results, "drift_alerts": alerts}


@app.get("/dashboard")
async def dashboard(model: str = DEFAULT_MODEL):
    summary = []
    for probe in PROBES:
        snaps = await get_snapshots(model, probe.id, limit=25)
        stats = compute_stats(snaps)
        recent_alerts = await get_alerts(model, limit=5)
        probe_alerts = [a for a in recent_alerts if a["probe_id"] == probe.id]
        summary.append({
            "probe_id": probe.id,
            "category": probe.category,
            "stats": stats,
            "recent_alerts": probe_alerts,
            "status": "critical" if any(a["severity"] == "critical" for a in probe_alerts)
            else "warning" if any(a["severity"] == "warning" for a in probe_alerts)
            else "healthy",
        })
    alerts = await get_alerts(model)
    return {"model": model, "probes": summary, "recent_alerts": alerts}


@app.get("/alerts")
async def list_alerts(model: str | None = None):
    return {"alerts": await get_alerts(model)}


@app.get("/health")
async def health():
    return {"status": "ok", "monitoring": DEFAULT_MODEL, "interval_seconds": MONITOR_INTERVAL}
