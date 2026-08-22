# 23 · Automated Retraining Pipeline

Monitors a deployed model's accuracy and latency against a live labeled dataset. When performance degrades below configurable thresholds, the pipeline automatically triggers retraining. Uses AG News (open-source) to simulate realistic data drift.

## How it works

```
live inferences
  │  log_inference(features, prediction, latency_ms)
  ▼
collector.py  ──► data/inference.db (SQLite)
  │  log_label(id, ground_truth)
  ▼
monitor.py    ──► accuracy, p50/p95 latency (sliding 24h window)
  │
  ▼
trigger.py    ──► should_retrain() → {retrain: bool, reasons: [...]}
  │
  ├── No degradation → sleep(interval)
  │
  └── Degradation detected
        │
        ▼
      trainer.py → retrain on labeled window → save model
```

## Quick start

```bash
pip install -e .

# Simulate clean data then drift, watch pipeline trigger
python scripts/simulate_drift.py --clean 60 --drift 60 --interval 0.05
```

Expected output:

```
Phase 1 — 60 clean samples (model performs well)
  logged 20/60
  logged 40/60
  logged 60/60

Phase 2 — 60 drift samples (model degrades)
  logged 20/60
  logged 40/60
  logged 60/60

Running pipeline check...
Decision: retrain=True
Reasons:  ['Accuracy 0.27% below threshold 70.00%', 'p95 latency 487ms exceeds threshold 300ms']
[pipeline] Retrained on 120 samples — reasons: [...]
```

## Configuration

All thresholds are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRAIN_INTERVAL_S` | 3600 | How often the pipeline checks (seconds) |
| `MIN_ACCURACY` | 0.85 | Trigger retraining below this accuracy |
| `MAX_LATENCY_P95_MS` | 500 | Trigger above this p95 latency |
| `MIN_LABELED_SAMPLES` | 50 | Don't decide with fewer samples than this |

## Eval dataset

The drift simulation uses **AG News** (4-class news classification, 120k train / 7.6k test) from HuggingFace datasets. It logs correct predictions in phase 1 (clean), then random predictions in phase 2 (drift) — simulating a model encountering out-of-distribution data.

```python
from datasets import load_dataset
ds = load_dataset("ag_news", split="test")
```

No API key required. Downloads automatically on first run.

## Run with Docker

```bash
docker-compose up
```

The `pipeline` service runs on a 1-hour check interval by default. The `api` service (port 8023) exposes endpoints for logging inferences and labels from your application:

```bash
# Log an inference from your app
curl -X POST http://localhost:8023/infer \
  -H "Content-Type: application/json" \
  -d '{"features": {"text": "Breaking news..."}, "prediction": "World"}'

# Log the ground-truth label when it arrives
curl -X POST http://localhost:8023/label/1 \
  -d '{"label": "World"}'

# Check current model health
curl http://localhost:8023/health
```

## K8s deployment

```bash
kubectl apply -f k8s/
```

The pipeline runs as a single-replica Deployment (no benefit from scaling). The SQLite database is on a PersistentVolumeClaim shared with the API service.

## Skills demonstrated

- Sliding-window performance monitoring using live inference data
- Threshold-based degradation detection (accuracy + latency)
- Data collection pipeline with SQLite and ground-truth label linking
- Automated retraining triggered by real signals, not a schedule
- Open-source eval dataset (AG News) for realistic drift simulation
- Clean separation: collect → monitor → trigger → train
- Docker Compose orchestration for multi-service ML pipeline
