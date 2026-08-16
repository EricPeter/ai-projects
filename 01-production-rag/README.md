# Production RAG with Citations

PDF Q&A API with page-number citations, hybrid search (dense + BM25), and cross-encoder reranking.

## Stack
- **LangGraph** — RAG pipeline as a state graph
- **SQLite-vec** — vector storage (embedded, no external service)
- **sentence-transformers** — embeddings + cross-encoder reranking
- **rank-bm25** — sparse keyword search
- **Anthropic claude-sonnet-4-6** — answer generation

## Prerequisites
- Python 3.11+, uv (`pip install uv`)
- Docker & Docker Compose (optional)
- Anthropic API key

## Setup

```bash
cd 01-production-rag
cp .env.example .env          # add your ANTHROPIC_API_KEY
uv sync
```

## Run

```bash
# Option A: directly
uv run uvicorn src.main:app --reload

# Option B: Docker
docker-compose up
```

API available at http://localhost:8000. Docs at http://localhost:8000/docs.

## Usage

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"

# Query with citations
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Response:
```json
{
  "answer": "The document covers...",
  "citations": [3, 7, 12],
  "sources": [
    {"page": 3, "content": "...", "score": 0.94}
  ]
}
```

## Architecture

```
Upload → chunk PDF → embed chunks → store in SQLite-vec + BM25 index
Query  → dense search + BM25 search → reciprocal rank fusion → cross-encoder rerank → LLM generate → return with citations
```
