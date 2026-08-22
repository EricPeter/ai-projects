# 21 · ONNX Model Serving

High-performance ML model serving with ONNX Runtime, Prometheus instrumentation, Triton-compatible configuration, and a CI/CD eval gate that blocks deployment when accuracy regresses.

## Architecture

```
client
  │
  ▼
FastAPI  (/infer, /benchmark, /metrics)
  │
  ▼
ONNXModel  (onnxruntime.InferenceSession)
  │
  ├── CPUExecutionProvider  (default)
  └── CUDAExecutionProvider (if GPU available)

Prometheus ──► Grafana dashboard
```

## Quick start

```bash
# Export distilbert to ONNX first
pip install -e ".[export]"
python scripts/export_onnx.py --output models/distilbert.onnx

# Start the stack (api + prometheus + grafana)
docker-compose up
```

| Service    | URL                        |
|------------|----------------------------|
| API        | http://localhost:8021       |
| Prometheus | http://localhost:9090       |
| Grafana    | http://localhost:3000       |

## Endpoints

| Method | Path                        | Description                              |
|--------|-----------------------------|------------------------------------------|
| GET    | `/health`                   | Liveness check + loaded model names      |
| GET    | `/models`                   | List all loaded models with I/O shapes   |
| GET    | `/models/{name}`            | Metadata for one model                   |
| POST   | `/infer/{model_name}`       | Run inference, returns outputs + latency |
| POST   | `/benchmark/{model_name}`   | p50/p95/p99 latency + throughput         |
| GET    | `/metrics`                  | Prometheus metrics                       |

## Inference

```bash
curl -X POST http://localhost:8021/infer/distilbert \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"input_ids": [[101,2023,2003,1037,4937,102]], "attention_mask": [[1,1,1,1,1,1]]}}'
```

```json
{"outputs": {"logits": [[-2.3, 2.8]]}, "latency_ms": 4.2}
```

## Benchmarking

```bash
curl -X POST http://localhost:8021/benchmark/distilbert \
  -H "Content-Type: application/json" \
  -d '{"warmup": 50, "iterations": 200}'
```

```json
{
  "p50_ms": 4.1,
  "p95_ms": 6.3,
  "p99_ms": 8.7,
  "mean_ms": 4.3,
  "throughput_rps": 231.4
}
```

## Eval on SST-2 (open-source benchmark)

```bash
pip install -e ".[eval]"
python scripts/eval.py --model models/distilbert.onnx --split validation
```

```
Dataset : glue/sst2/validation  (872 samples)
Accuracy: 0.9106  (794/872)
Latency : p50=4.1ms  p95=6.2ms
```

The CI pipeline runs this automatically and blocks merges if accuracy drops below 90%.

## Triton Inference Server

The `triton/config.pbtxt` configures the model for NVIDIA Triton with dynamic batching:

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton:/models/distilbert \
  -v $(pwd)/models/distilbert.onnx:/models/distilbert/1/model.onnx \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

## K8s deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Port-forward for local access
kubectl port-forward svc/model-serving 8021:8021

# Check HPA scaling
kubectl get hpa model-serving --watch
```

## Skills demonstrated

- ONNX Runtime inference with graph optimisation and execution provider selection
- Dynamic batching configuration for throughput vs latency trade-offs
- Prometheus custom metrics: inference latency histograms, request counters
- CI/CD accuracy gate using a real open-source eval dataset (SST-2)
- Triton Inference Server model repository config
- Kubernetes Deployment + HPA for CPU-driven autoscaling
- Multi-stage Docker build to minimise image size
