import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from .load_balancer import get_best_instance, health_check_loop, get_instances_status

requests_total = Counter("inference_requests_total", "Total requests", ["status"])
latency_hist = Histogram("inference_latency_seconds", "Latency", buckets=[0.1, 0.5, 1, 2, 5, 10, 30])
ttft_hist = Histogram("inference_ttft_seconds", "Time to first token", buckets=[0.05, 0.1, 0.5, 1, 2, 5])


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(health_check_loop())
    yield


app = FastAPI(title="Production Inference Proxy", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "facebook/opt-125m"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    instance = await get_best_instance()
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{instance}/v1/chat/completions",
                json=body.model_dump(),
            )
            r.raise_for_status()
    except Exception as e:
        requests_total.labels(status="error").inc()
        raise HTTPException(502, f"vLLM error: {e}")

    elapsed = time.perf_counter() - start
    latency_hist.observe(elapsed)
    requests_total.labels(status="success").inc()
    return r.json()


@app.get("/v1/models")
async def list_models():
    instance = await get_best_instance()
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{instance}/v1/models")
    return r.json()


@app.get("/health")
def health():
    return {"status": "ok", "instances": get_instances_status()}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
