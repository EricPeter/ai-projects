# Vector Database at Scale

Qdrant-powered vector search with hybrid retrieval, embedding caching (Redis), metadata filtering, and automated backups.

## Features
- **Hybrid search** — dense vectors + Reciprocal Rank Fusion
- **Embedding cache** — Redis with 24h TTL, keyed by content hash
- **Metadata filters** — equality, range, and contains queries
- **Index optimisation** — configurable HNSW params via API
- **QPS benchmark** — measures latency at k=1, 5, 10, 50

## Setup

```bash
cd 12-vector-db
cp .env.example .env
docker-compose up
```

## Usage

```bash
# Create a collection
curl -X POST http://localhost:8012/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "vector_size": 384}'

# Ingest documents
curl -X POST http://localhost:8012/ingest \
  -H "Content-Type: application/json" \
  -d '{"collection": "docs", "documents": [{"text": "RAG systems combine retrieval with generation", "category": "ai"}]}'

# Search with metadata filter
curl -X POST http://localhost:8012/search \
  -H "Content-Type: application/json" \
  -d '{"collection": "docs", "query": "retrieval augmented generation", "k": 5, "metadata_filter": {"category": "ai"}}'

# Benchmark
uv run python scripts/benchmark.py
```
