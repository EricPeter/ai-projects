from __future__ import annotations

import random
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time

from .config import settings
from .metrics import ab_requests_total, ab_traffic_split, store

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(timeout=10.0)
    ab_traffic_split.set(settings.traffic_split)
    yield
    await _client.aclose()


app = FastAPI(title="A/B Model Router", lifespan=lifespan)


def _pick_variant() -> tuple[str, str]:
    if random.random() < settings.traffic_split:
        return "b", settings.model_b_url
    return "a", settings.model_a_url


@app.get("/health")
def health():
    return {"status": "ok", "traffic_split": settings.traffic_split}


@app.post("/infer")
async def infer(request: Request):
    variant, url = _pick_variant()
    body = await request.body()
    t0 = time.perf_counter()
    try:
        resp = await _client.post(f"{url}/infer", content=body, headers=dict(request.headers))
        latency = time.perf_counter() - t0
        success = resp.status_code < 400
        store.record(variant, latency, success)
        ab_requests_total.labels(variant=variant, status_code=str(resp.status_code)).inc()
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
            headers={"X-Model-Variant": variant},
        )
    except httpx.RequestError:
        store.record(variant, time.perf_counter() - t0, False)
        ab_requests_total.labels(variant=variant, status_code="503").inc()
        return JSONResponse({"error": f"model-{variant} unavailable"}, status_code=503)


@app.get("/experiment")
def experiment():
    return store.get_summary(
        min_samples=settings.min_samples,
        alpha=settings.alpha,
    )


@app.post("/experiment/reset")
def reset():
    store.reset()
    return {"status": "reset"}


@app.put("/experiment/split")
def update_split(body: dict):
    split = float(body.get("split", settings.traffic_split))
    split = max(0.0, min(1.0, split))
    settings.traffic_split = split
    ab_traffic_split.set(split)
    return {"traffic_split": split}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
