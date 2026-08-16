# AI Engineering Portfolio

20 production-grade AI systems demonstrating real-world engineering skills. Every project is runnable end-to-end.

## Projects

| # | Project | Stack | Run |
|---|---------|-------|-----|
| 01 | [Production RAG with Citations](./01-production-rag/) | LangGraph + SQLite-vec + cross-encoder | `docker-compose up` |
| 02 | [Cost-Optimized Model Router](./02-model-router/) | Anthropic SDK + Prometheus | `docker-compose up` |
| 03 | [Multi-Agent Research System](./03-multi-agent-research/) | CrewAI | `uv run python -m src.main` |
| 04 | [Automated Eval Harness](./04-eval-harness/) | DeepEval + RAGAS + LangSmith | `uv run pytest` |
| 05 | [Real-Time Observability Dashboard](./05-observability/) | OpenTelemetry + Grafana + Prometheus | `docker-compose up` |
| 06 | [Security Guardrail Middleware](./06-security-guardrails/) | Presidio + FastAPI | `uv run uvicorn src.main:app` |
| 07 | [Local-First Dev Environment](./07-local-dev/) | Ollama + LanceDB + FastAPI | `docker-compose up` |
| 08 | [Streaming Copilot UI](./08-streaming-copilot/) | Next.js 14 + Vercel AI SDK | `npm run dev` |
| 09 | [Fine-Tuning Pipeline with LoRA](./09-fine-tuning/) | PEFT + Hugging Face TRL | `bash scripts/run_pipeline.sh` |
| 10 | [Multi-Tenant SaaS Agent](./10-multi-tenant-saas/) | FastAPI + SQLAlchemy + Stripe | `uv run uvicorn src.main:app` |
| 11 | [CI/CD for AI Systems](./11-cicd-ai/) | GitHub Actions + ArgoCD + Kubernetes | See README |
| 12 | [Vector Database at Scale](./12-vector-db/) | Qdrant + Redis | `docker-compose up` |
| 13 | [Agent Memory System](./13-agent-memory/) | Redis + Qdrant | `docker-compose up` |
| 14 | [Production Inference Server](./14-inference-server/) | vLLM + Kubernetes | `docker-compose up` |
| 15 | [Human-in-the-Loop Workflow](./15-hitl-workflow/) | LangGraph + FastAPI + WebSocket | `uv run uvicorn src.main:app` |
| 16 | [Agentic Automation Pipeline](./16-agentic-automation/) | FastAPI + Celery + Redis | `docker-compose up` |
| 17 | [Legal AI Benchmark](./17-domain-benchmark/) | DeepEval + sentence-transformers | `uv run pytest` |
| 18 | [Sunflower Multilingual API](./18-sunflower-translator/) | Sunflower-14B + vLLM + FastAPI | `docker-compose up` |
| 19 | [Uganda Community Health Bot](./19-uganda-health-bot/) | Sunflower-14B + FastAPI + Twilio | `uv run uvicorn src.main:app` |
| 20 | [Model Drift Monitor](./20-model-drift-monitor/) | sentence-transformers + scipy + SQLite | `docker-compose up` |
| — | [Interactive Playground](./playground/) | Next.js 14 + Tailwind | `npm run dev` |

## Quick Start

Each project is self-contained. Navigate into any project directory and follow its README.

```bash
# Prerequisites
pip install uv        # Python package manager
# Docker Desktop      # for projects with docker-compose.yml

# General pattern for Python projects
cd <project-dir>
cp .env.example .env  # add your API keys
uv sync
uv run uvicorn src.main:app --reload   # or: docker-compose up
```

## Stack Overview

- **Languages:** Python 3.11, TypeScript
- **Package management:** uv (Python), npm (TypeScript)
- **LLM APIs:** Anthropic Claude (primary), Sunbird Sunflower-14B (African languages)
- **Infrastructure:** Docker, Kubernetes, GitHub Actions, ArgoCD
- **Databases:** SQLite, Redis, Qdrant, LanceDB, PostgreSQL
- **Observability:** OpenTelemetry, Prometheus, Grafana, Jaeger
