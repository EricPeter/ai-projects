# Cost-Optimized Model Router

Routes queries to the cheapest model that can handle them. Tracks spend per request. Exposes Prometheus metrics.

## Routing Logic

| Complexity | Criteria | Model |
|-----------|---------|-------|
| Simple | <30 words, no code, no analysis | claude-haiku (cheapest) |
| Medium | General questions, moderate length | claude-sonnet |
| Complex | Code, math, multi-step, >150 words | claude-opus (most capable) |

## Setup

```bash
cd 02-model-router
cp .env.example .env    # add ANTHROPIC_API_KEY
uv sync
uv run uvicorn src.main:app --reload
```

## Usage

```bash
# Simple query → routes to Haiku
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'

# Complex query → routes to Opus
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Implement a red-black tree in Python with full insert and delete operations, analyzing the time complexity of each operation."}'

# View cost breakdown
curl http://localhost:8000/costs

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Docker (includes Prometheus)

```bash
docker-compose up
# Prometheus UI: http://localhost:9090
```
