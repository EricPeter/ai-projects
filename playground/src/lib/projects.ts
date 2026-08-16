export type PlaygroundType =
  | "chat"
  | "text-in-json-out"
  | "rag"
  | "translate"
  | "search"
  | "eval"
  | "infra";

export interface Project {
  id: number;
  slug: string;
  name: string;
  description: string;
  tech: string[];
  port: number;
  playgroundType: PlaygroundType;
  endpoints?: Record<string, string>;
  dashboards?: { label: string; url: string }[];
}

export const PROJECTS: Project[] = [
  {
    id: 1,
    slug: "production-rag",
    name: "Production RAG",
    description: "Hybrid search (dense + BM25) with cross-encoder reranking, page citations, and LangGraph orchestration.",
    tech: ["LangGraph", "SQLite-vec", "sentence-transformers", "FastAPI"],
    port: 8001,
    playgroundType: "rag",
    endpoints: { ingest: "/ingest", query: "/query" },
  },
  {
    id: 2,
    slug: "model-router",
    name: "Model Router",
    description: "Routes queries to Haiku, Sonnet, or Opus based on heuristic complexity analysis with cost estimation.",
    tech: ["Anthropic", "FastAPI", "Heuristics"],
    port: 8002,
    playgroundType: "text-in-json-out",
    endpoints: { route: "/route" },
  },
  {
    id: 3,
    slug: "multi-agent-research",
    name: "Multi-Agent Research",
    description: "CrewAI research pipeline with planner, researcher, writer, and editor agents with JSONL audit trails.",
    tech: ["CrewAI", "Anthropic", "FastAPI"],
    port: 8003,
    playgroundType: "text-in-json-out",
    endpoints: { research: "/research" },
  },
  {
    id: 4,
    slug: "eval-harness",
    name: "Eval Harness",
    description: "Automated evaluation with DeepEval metrics — answer relevancy, hallucination, ROUGE-L, semantic similarity.",
    tech: ["DeepEval", "Anthropic", "FastAPI"],
    port: 8004,
    playgroundType: "eval",
    endpoints: { evaluate: "/evaluate" },
  },
  {
    id: 5,
    slug: "observability",
    name: "Observability Stack",
    description: "OpenTelemetry traces, Prometheus metrics, and Grafana dashboards for AI inference monitoring.",
    tech: ["OpenTelemetry", "Prometheus", "Grafana", "Jaeger"],
    port: 8005,
    playgroundType: "infra",
    dashboards: [
      { label: "Grafana", url: "http://localhost:3000" },
      { label: "Jaeger", url: "http://localhost:16686" },
      { label: "Prometheus", url: "http://localhost:9090" },
    ],
    endpoints: { complete: "/complete" },
  },
  {
    id: 6,
    slug: "security-guardrails",
    name: "Security Guardrails",
    description: "PII detection and anonymization, prompt injection detection, and output safety filtering.",
    tech: ["Presidio", "Anthropic", "FastAPI"],
    port: 8006,
    playgroundType: "text-in-json-out",
    endpoints: { check: "/check" },
  },
  {
    id: 7,
    slug: "local-dev",
    name: "Local Dev Assistant",
    description: "Chat with local Ollama models + LanceDB semantic memory. Fully private, runs offline.",
    tech: ["Ollama", "LanceDB", "FastAPI"],
    port: 8007,
    playgroundType: "chat",
    endpoints: { chat: "/chat" },
  },
  {
    id: 8,
    slug: "streaming-copilot",
    name: "Streaming Copilot",
    description: "Next.js streaming chat UI with Vercel AI SDK, token-level streaming, and markdown rendering.",
    tech: ["Next.js", "Vercel AI SDK", "Anthropic"],
    port: 8008,
    playgroundType: "infra",
    dashboards: [{ label: "Open Copilot", url: "http://localhost:8008" }],
  },
  {
    id: 9,
    slug: "fine-tuning",
    name: "Fine-Tuning Pipeline",
    description: "LoRA fine-tuning with SFT + DPO, catastrophic forgetting detection, and ROUGE-L evaluation.",
    tech: ["PEFT", "TRL", "HuggingFace", "FastAPI"],
    port: 8009,
    playgroundType: "infra",
    endpoints: { status: "/status", metrics: "/metrics" },
    dashboards: [{ label: "Training Metrics", url: "http://localhost:8009/metrics" }],
  },
  {
    id: 10,
    slug: "multi-tenant-saas",
    name: "Multi-Tenant SaaS",
    description: "JWT auth, row-level tenant isolation in PostgreSQL, and per-tenant rate limiting.",
    tech: ["FastAPI", "SQLAlchemy", "PostgreSQL", "JWT"],
    port: 8010,
    playgroundType: "text-in-json-out",
    endpoints: { complete: "/api/v1/complete" },
  },
  {
    id: 11,
    slug: "cicd-ai",
    name: "CI/CD + AI Eval Gate",
    description: "GitHub Actions pipeline with automated eval gate that blocks merges if quality score drops below 90%.",
    tech: ["GitHub Actions", "ArgoCD", "Kubernetes", "FastAPI"],
    port: 8011,
    playgroundType: "infra",
    dashboards: [{ label: "Pipeline Status", url: "http://localhost:8011/status" }],
  },
  {
    id: 12,
    slug: "vector-db",
    name: "Vector DB + Caching",
    description: "Qdrant hybrid search with Reciprocal Rank Fusion and Redis embedding cache (24h TTL).",
    tech: ["Qdrant", "Redis", "FastAPI"],
    port: 8012,
    playgroundType: "search",
    endpoints: { search: "/search", upsert: "/upsert" },
  },
  {
    id: 13,
    slug: "agent-memory",
    name: "Agent Memory",
    description: "Conversational agent with Redis sliding window (recent turns) + Qdrant semantic long-term recall.",
    tech: ["Redis", "Qdrant", "Anthropic", "FastAPI"],
    port: 8013,
    playgroundType: "chat",
    endpoints: { chat: "/chat" },
  },
  {
    id: 14,
    slug: "inference-server",
    name: "Inference Load Balancer",
    description: "Proxy that routes to the least-loaded vLLM replica with Prometheus TTFT/throughput metrics.",
    tech: ["vLLM", "Prometheus", "FastAPI"],
    port: 8014,
    playgroundType: "infra",
    dashboards: [{ label: "Load Balancer Status", url: "http://localhost:8014/status" }],
    endpoints: { complete: "/v1/chat/completions" },
  },
  {
    id: 15,
    slug: "hitl-workflow",
    name: "Human-in-the-Loop",
    description: "Detects irreversible or high-risk AI actions, pauses for human approval via WebSocket real-time UI.",
    tech: ["FastAPI", "WebSockets", "SQLite"],
    port: 8015,
    playgroundType: "text-in-json-out",
    endpoints: { submit: "/submit", approve: "/approve/{task_id}" },
  },
  {
    id: 16,
    slug: "agentic-automation",
    name: "Agentic Automation",
    description: "Celery async pipeline with exponential backoff, Redis idempotency keys, and Flower monitoring.",
    tech: ["Celery", "Redis", "FastAPI", "Flower"],
    port: 8016,
    playgroundType: "infra",
    dashboards: [{ label: "Flower Monitor", url: "http://localhost:5555" }],
    endpoints: { trigger: "/trigger", status: "/status/{task_id}" },
  },
  {
    id: 17,
    slug: "domain-benchmark",
    name: "Domain Benchmark",
    description: "Run models against domain-specific datasets, compute ROUGE-L + semantic similarity, publish leaderboard.",
    tech: ["FastAPI", "Anthropic", "sentence-transformers"],
    port: 8017,
    playgroundType: "eval",
    endpoints: { run: "/run", leaderboard: "/leaderboard" },
  },
  {
    id: 18,
    slug: "sunflower-translator",
    name: "Sunflower Translator",
    description: "Neural translation across 31 Ugandan languages using Sunbird Sunflower-14B-W8A8 via vLLM.",
    tech: ["Sunbird Sunflower-14B", "vLLM", "FastAPI"],
    port: 8018,
    playgroundType: "translate",
    endpoints: { translate: "/translate", languages: "/languages" },
  },
  {
    id: 19,
    slug: "uganda-health-bot",
    name: "Uganda Health Bot",
    description: "Community health worker in Luganda, Acholi, and English. Emergency detection with instant referral.",
    tech: ["Sunbird Sunflower-14B", "vLLM", "FastAPI"],
    port: 8019,
    playgroundType: "chat",
    endpoints: { respond: "/respond" },
  },
  {
    id: 20,
    slug: "model-drift-monitor",
    name: "Model Drift Monitor",
    description: "Continuously probes a model with canonical prompts and detects semantic, behavioral, and latency drift using cosine distance + KS statistics.",
    tech: ["sentence-transformers", "scipy", "SQLite", "FastAPI"],
    port: 8020,
    playgroundType: "eval",
    endpoints: { probe: "/probe", dashboard: "/dashboard", alerts: "/alerts" },
  },
];

export function getProject(id: number): Project | undefined {
  return PROJECTS.find((p) => p.id === id);
}

export function projectApiUrl(port: number, path: string): string {
  const base = process.env.NEXT_PUBLIC_API_BASE || `http://localhost:${port}`;
  return `${base}${path}`;
}
