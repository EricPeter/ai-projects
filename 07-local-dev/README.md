# Local-First Development Environment

Zero API cost RAG system using Ollama + LanceDB + FastAPI. Mirrors production architecture, runs 100% offline.

## Stack
- **Ollama** — local LLM inference (llama3.2)
- **LanceDB** — embedded vector database (no server required)
- **sentence-transformers** — local embeddings
- **FastAPI** — REST API

## Setup

```bash
cd 07-local-dev
cp .env.example .env
uv sync

# Pull the model (requires Ollama installed locally)
ollama pull llama3.2
uv run uvicorn src.main:app --reload
```

## Docker (recommended)

```bash
docker-compose up
# First run pulls the model automatically (~2GB)
```

## Usage

```bash
# Upload a file
curl -X POST http://localhost:8007/upload \
  -F "file=@your_doc.pdf"

# Query
curl -X POST http://localhost:8007/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'

# Run full demo
uv run python scripts/demo.py
```
