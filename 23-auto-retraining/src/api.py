from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .collector import log_inference, log_label, labeled_count
from .monitor import health_report
from .trigger import Thresholds, should_retrain

app = FastAPI()
_thresholds = Thresholds()


class InferRequest(BaseModel):
    features: dict
    prediction: str
    latency_ms: float = 0.0


class LabelRequest(BaseModel):
    label: str


@app.post("/infer")
def infer(req: InferRequest):
    inference_id = log_inference(req.features, req.prediction, req.latency_ms)
    return {"id": inference_id}


@app.post("/label/{inference_id}")
def label(inference_id: int, req: LabelRequest):
    log_label(inference_id, req.label)
    return {"ok": True}


@app.get("/health")
def health():
    report = health_report()
    decision = should_retrain(_thresholds)
    return {
        "labeled_last_24h": labeled_count(),
        "accuracy": report.get("accuracy"),
        "latency": report.get("latency"),
        "retrain_needed": decision["retrain"],
        "reasons": decision["reasons"],
    }
