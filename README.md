# AI Engineering Portfolio

17 production-grade AI systems demonstrating real-world engineering skills.

## Projects

| # | Project | Stack | Run |
|---|---------|-------|-----|
| 01 | [Production RAG with Citations](./01-production-rag/) | LangGraph + SQLite-vec + cross-encoder | `docker-compose up` |
| 02 | [Cost-Optimized Model Router](./02-model-router/) | LiteLLM + Prometheus | `docker-compose up` |
| 03 | [Multi-Agent Research System](./03-multi-agent-research/) | CrewAI | `uv run python main.py` |
| 04 | [Automated Eval Harness](./04-eval-harness/) | DeepEval + RAGAS + LangSmith | `uv run pytest` |
| 05 | [Real-Time Observability Dashboard](./05-observability/) | OpenTelemetry + Grafana + Prometheus | `docker-compose up` |
| 06 | [Security Guardrail Middleware](./06-security-guardrails/) | Guardrails AI + FastAPI | `docker-compose up` |
| 07 | [Local-First Dev Environment](./07-local-dev/) | Ollama + LanceDB + FastAPI | `docker-compose up` |
| 08 | [Streaming Copilot UI](./08-streaming-copilot/) | Next.js + Vercel AI SDK | `npm run dev` |
| 09 | [Fine-Tuning Pipeline with LoRA](./09-fine-tuning/) | PEFT + Hugging Face | `uv run python train.py` |
| 10 | [Multi-Tenant SaaS Agent](./10-multi-tenant-saas/) | Supabase + Stripe + LangGraph | `docker-compose up` |
| 11 | [CI/CD for AI Systems](./11-cicd-ai/) | GitHub Actions + ArgoCD | See README |
| 12 | [Vector Database at Scale](./12-vector-db/) | Qdrant | `docker-compose up` |
| 13 | [Agent Memory System](./13-agent-memory/) | Redis + Qdrant | `docker-compose up` |
| 14 | [Production Inference Server](./14-inference-server/) | vLLM + Kubernetes | `docker-compose up` |
| 15 | [Human-in-the-Loop Workflow](./15-hitl-workflow/) | LangGraph + FastAPI UI | `docker-compose up` |
| 16 | [Agentic Automation Pipeline](./16-agentic-automation/) | FastAPI + Celery | `docker-compose up` |
| 17 | [Domain-Specific Benchmark](./17-domain-benchmark/) | DeepEval + pytest | `uv run pytest` |

## Quick Start

Each project is self-contained. Navigate into any project and follow its README.

**Prerequisites:** Docker, Python 3.11+, uv (`pip install uv`)
