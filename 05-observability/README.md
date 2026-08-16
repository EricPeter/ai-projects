# Real-Time Observability Dashboard

Full-stack AI observability: distributed tracing, Prometheus metrics, and a pre-built Grafana dashboard.

## What you get
- **Distributed tracing** via OpenTelemetry → Jaeger
- **Metrics**: request rate, latency (p50/p95/p99), cost/hour, error rate
- **Grafana dashboard** pre-provisioned and auto-loaded
- **Cost tracking** per model with Anthropic pricing

## Setup

```bash
cd 05-observability
cp .env.example .env   # add ANTHROPIC_API_KEY
docker-compose up
```

| Service | URL |
|---------|-----|
| API | http://localhost:8005 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Jaeger traces | http://localhost:16686 |

## Usage

```bash
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum entanglement simply."}'
```

Then open Grafana at http://localhost:3000 → Dashboards → AI Proxy.
