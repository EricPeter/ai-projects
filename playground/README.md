# AI Projects Playground

Interactive frontend for all 20 AI engineering projects. Each project gets a purpose-built UI — chat interfaces, RAG with PDF upload, semantic search, translation, eval dashboards, drift monitoring, and infrastructure status panels.

## Run locally

```bash
npm install
npm run dev
```

Opens on http://localhost:3020

## Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Connects to project backends on ports 8001–8020

## Per-project UIs

| Project | UI type |
|---------|---------|
| 01 Production RAG | PDF upload + question answering with citations |
| 02 Model Router | Text input → model selection + reasoning |
| 03 Multi-Agent Research | Topic input → research report |
| 06 Security Guardrails | PII and injection detection results |
| 07 Local Dev | Chat (Ollama) |
| 10 Multi-Tenant SaaS | Prompt → completion |
| 12 Vector DB | Semantic search + document insertion |
| 13 Agent Memory | Chat with persistent memory |
| 15 HITL Workflow | Action risk evaluation |
| 17 Domain Benchmark | Leaderboard |
| 18 Sunflower Translator | Side-by-side 31-language translation |
| 19 Uganda Health Bot | Multi-language chat with emergency detection |
| 20 Model Drift Monitor | Live probe runner + drift alert dashboard |
| Infrastructure projects (5, 8, 9, 11, 14, 16) | Dashboard links + status checks |

## Docker

```bash
docker compose up
```

Runs on http://localhost:3020
