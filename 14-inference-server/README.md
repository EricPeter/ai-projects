# Production Inference Server

vLLM load-balanced proxy with health checking, Prometheus metrics, and automatic failover.

## Architecture

```
Client → Smart Proxy (FastAPI)
            ↓  least-queue routing
         vLLM-1   vLLM-2
         (health check every 10s)
```

## Setup

```bash
cd 14-inference-server
cp .env.example .env
docker-compose up
```

The first run downloads the model (~250MB for opt-125m). Swap for a larger model in `docker-compose.yml`.

## Usage

```bash
curl -X POST http://localhost:8014/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Instance health
curl http://localhost:8014/health

# Prometheus metrics
curl http://localhost:8014/metrics
```

## Production Notes

- Swap `facebook/opt-125m` for `meta-llama/Llama-3-8B-Instruct` (requires GPU + HF token)
- Add `--quantization awq` to vLLM command for 4-bit quantization
- Use `--tensor-parallel-size N` for multi-GPU setups
