# 20 · Model Drift Monitor

Continuously probes a model with a fixed set of canonical prompts and detects performance drift across three dimensions: semantic consistency, keyword hit rate, and latency percentiles. Raises `warning` or `critical` alerts when drift exceeds thresholds.

## How it works

1. **Probes** — 6 canonical prompts across factual, reasoning, safety, instruction-following, and coding categories
2. **Snapshots** — every response is embedded (MiniLM-L6-v2) and stored in SQLite with latency and token counts
3. **Drift detection** — compares the 5 most-recent responses against a 20-snapshot baseline using:
   - **Semantic drift**: cosine distance between centroid embeddings
   - **Keyword hit rate**: fraction of expected keywords present in the response
   - **Latency P95**: relative change in 95th-percentile inference time
4. **Alerts** — stored in SQLite, exposed via `/alerts` and `/dashboard`

## Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Semantic similarity (cosine delta) | 0.08 | 0.15 |
| Keyword hit rate drop | 10% | 20% |
| Latency P95 increase | 30% | 60% |

## Run

```bash
cp .env.example .env
# edit ANTHROPIC_API_KEY
docker compose up
```

API on http://localhost:8020

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/probe` | Run all probes now, get immediate results + any drift alerts |
| `GET` | `/dashboard` | Full status: per-probe stats, recent alerts, health |
| `GET` | `/alerts` | All stored drift alerts |
| `GET` | `/health` | Service health |

## Run tests

```bash
uv run pytest tests/
```
