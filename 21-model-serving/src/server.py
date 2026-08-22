from __future__ import annotations

import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .model import ONNXModel

# ── Prometheus metrics ────────────────────────────────────────────────────────
inference_duration = Histogram(
    "model_inference_duration_seconds",
    "Inference latency",
    labelnames=["model_name", "batch_size"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
requests_total = Counter(
    "model_requests_total",
    "Total inference requests",
    labelnames=["model_name", "status"],
)
batch_size_hist = Histogram(
    "model_batch_size",
    "Batch sizes seen at inference",
    labelnames=["model_name"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

# ── Model registry ────────────────────────────────────────────────────────────
_models: dict[str, ONNXModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    if model_dir.exists():
        for onnx_file in model_dir.glob("*.onnx"):
            name = onnx_file.stem
            print(f"Loading model: {name} from {onnx_file}")
            _models[name] = ONNXModel(str(onnx_file))
    if not _models:
        print("No .onnx files found — start with MODEL_DIR pointing to a directory with .onnx files.")
    yield
    _models.clear()


app = FastAPI(title="ONNX Model Server", version="0.1.0", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": list(_models)}


@app.get("/models")
def list_models():
    return [{"name": name, **model.metadata} for name, model in _models.items()]


@app.get("/models/{name}")
def get_model(name: str):
    if name not in _models:
        raise HTTPException(404, f"Model '{name}' not loaded")
    return {"name": name, **_models[name].metadata}


@app.post("/infer/{model_name}")
def infer(model_name: str, body: dict):
    """Run inference. Body: {"inputs": {"input_name": [[values]]}}"""
    if model_name not in _models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    model = _models[model_name]
    raw = body.get("inputs", {})
    inputs = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    batch = next(iter(inputs.values())).shape[0] if inputs else 1

    t0 = time.perf_counter()
    try:
        outputs = model.run(inputs)
        latency = time.perf_counter() - t0
        inference_duration.labels(model_name=model_name, batch_size=str(batch)).observe(latency)
        batch_size_hist.labels(model_name=model_name).observe(batch)
        requests_total.labels(model_name=model_name, status="success").inc()
        return {
            "outputs": {k: v.tolist() for k, v in outputs.items()},
            "latency_ms": round(latency * 1000, 3),
        }
    except Exception as exc:
        requests_total.labels(model_name=model_name, status="error").inc()
        raise HTTPException(500, str(exc)) from exc


@app.post("/benchmark/{model_name}")
def benchmark(model_name: str, body: dict | None = None):
    """Run a latency benchmark. Body: {"inputs": {...}, "warmup": 50, "iterations": 200}"""
    if model_name not in _models:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    model = _models[model_name]
    body = body or {}
    warmup = body.get("warmup", 50)
    iterations = body.get("iterations", 200)

    # Build random inputs matching the model's expected shapes
    raw = body.get("inputs")
    if raw:
        inputs = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    else:
        inputs = {}
        for inp in model.session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    return model.benchmark(inputs, warmup=warmup, iterations=iterations)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
