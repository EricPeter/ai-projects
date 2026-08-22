# 22 · A/B Model Rollout Infrastructure

Traffic-splitting proxy for controlled ML model rollouts. Routes a configurable fraction of requests to a challenger model, collects per-variant latency and success-rate metrics, and runs Welch t-tests + two-proportion z-tests to determine when a winner is statistically significant.

## How it works

```
client
  │
  ▼
router  (POST /infer)
  │
  ├── 90% ──► model-a  (champion)
  └── 10% ──► model-b  (challenger)
       │
       ▼
  ExperimentStore
  (per-variant latency samples, success counts)
       │
       ▼
  Welch t-test  +  two-proportion z-test
       │
       ▼
  GET /experiment  →  winner, p-value, effect %
```

## Quick start

```bash
docker-compose up
```

| Service    | URL                   |
|------------|-----------------------|
| Router     | http://localhost:8022 |
| Prometheus | http://localhost:9090 |
| Grafana    | http://localhost:3000 |

The `model-a` and `model-b` containers are stubs: model-b has mean latency 80ms vs model-a's 100ms, so it should win after ~200 requests.

## Send traffic

```bash
# Single request
curl -X POST http://localhost:8022/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'

# Load test (500 requests)
for i in $(seq 500); do
  curl -s -X POST http://localhost:8022/infer -H "Content-Type: application/json" \
    -d '{"text": "sample"}' > /dev/null
done
```

The response header `X-Model-Variant: a|b` tells you which model handled the request.

## Check experiment results

```bash
curl http://localhost:8022/experiment | jq .
```

```json
{
  "variants": {
    "a": {"mean": 101.2, "p95": 148.3, "n": 450},
    "b": {"mean":  81.4, "p95": 118.6, "n":  50}
  },
  "latency_test": {
    "test": "welch_t",
    "p_value": 0.0002,
    "significant": true,
    "winner": "b",
    "effect_pct": -19.6
  },
  "significant": true,
  "recommendation": "Promote model b"
}
```

## Dynamic traffic shifting

Gradually increase split as confidence grows:

```bash
# Start at 10%
curl -X PUT http://localhost:8022/experiment/split -d '{"split": 0.10}'

# After significance: move to 50%
curl -X PUT http://localhost:8022/experiment/split -d '{"split": 0.50}'

# Full rollout
curl -X PUT http://localhost:8022/experiment/split -d '{"split": 1.00}'

# Reset if needed
curl -X POST http://localhost:8022/experiment/reset
```

## Statistical tests

**Welch t-test** — compares latency distributions without assuming equal variance. P-value < 0.05 + minimum sample count → significant.

**Two-proportion z-test** — compares success rates (non-5xx responses) between variants. Catches cases where a faster model also has higher error rates.

Both tests run simultaneously; either reaching significance triggers a recommendation.

## Prometheus metrics

| Metric | Type | Labels |
|--------|------|--------|
| `ab_requests_total` | counter | variant, status_code |
| `ab_latency_seconds` | histogram | variant |
| `ab_experiment_traffic_split` | gauge | — |
| `ab_experiment_samples_total` | gauge | variant |
| `ab_significance_achieved` | gauge | — |

## K8s deployment

```bash
kubectl apply -f k8s/
kubectl port-forward svc/ab-router-svc 8022:8022
```

## Skills demonstrated

- Traffic splitting with configurable, runtime-adjustable percentages
- Welch t-test and two-proportion z-test for statistical significance
- Thread-safe in-memory metric collection with sliding window (max 10,000 samples)
- Prometheus custom gauges, counters, histograms
- httpx async proxying with error handling and per-variant attribution
- Dynamic config via PUT endpoint — no restart required
- K8s-ready with readiness probes and Prometheus scrape annotations
