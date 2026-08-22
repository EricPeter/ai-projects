import asyncio
import os
import random

from fastapi import FastAPI

app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "model-a")
LATENCY_MEAN = float(os.environ.get("MODEL_LATENCY_MEAN", "0.1"))
LATENCY_STD = float(os.environ.get("MODEL_LATENCY_STD", "0.02"))
ERROR_RATE = float(os.environ.get("MODEL_ERROR_RATE", "0.02"))


@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID}


@app.post("/infer")
async def infer():
    latency = max(0, random.gauss(LATENCY_MEAN, LATENCY_STD))
    await asyncio.sleep(latency)
    if random.random() < ERROR_RATE:
        from fastapi import HTTPException
        raise HTTPException(500, "simulated model error")
    return {
        "prediction": random.random(),
        "confidence": random.uniform(0.5, 1.0),
        "model_id": MODEL_ID,
    }
