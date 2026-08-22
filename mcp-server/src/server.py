from mcp.server.fastmcp import FastMCP
from . import health_bot, router

mcp = FastMCP(
    name="ai-projects",
    instructions=(
        "Tools from Eric Wairagala's AI engineering portfolio. "
        "The flagship tool is `ask_health_bot` — a community health assistant "
        "for Uganda that responds in Luganda, Acholi, Ateso, Runyankole, or English."
    ),
)

PROJECTS = [
    {"id": 1,  "name": "Production RAG with Citations",      "tech": "LangGraph, SQLite-vec, BM25",          "port": 8001},
    {"id": 2,  "name": "Cost-Optimized Model Router",        "tech": "Anthropic SDK, Prometheus",            "port": 8002},
    {"id": 3,  "name": "Multi-Agent Research System",        "tech": "CrewAI, Anthropic SDK",                "port": 8003},
    {"id": 4,  "name": "Automated Eval Harness",             "tech": "DeepEval, RAGAS, pytest",              "port": 8004},
    {"id": 5,  "name": "Real-Time Observability Dashboard",  "tech": "OpenTelemetry, Prometheus, Grafana",   "port": 8005},
    {"id": 6,  "name": "Security Guardrail Middleware",      "tech": "Presidio, FastAPI",                    "port": 8006},
    {"id": 7,  "name": "Local-First Dev Environment",        "tech": "Ollama, LanceDB",                      "port": 8007},
    {"id": 8,  "name": "Streaming Copilot UI",               "tech": "Next.js 14, Vercel AI SDK",            "port": 8008},
    {"id": 9,  "name": "Fine-Tuning Pipeline with LoRA",     "tech": "PEFT, TRL, Transformers",              "port": 8009},
    {"id": 10, "name": "Multi-Tenant SaaS Agent",            "tech": "FastAPI, SQLAlchemy, JWT",             "port": 8010},
    {"id": 11, "name": "CI/CD for AI Systems",               "tech": "GitHub Actions, ArgoCD, Kubernetes",   "port": 8011},
    {"id": 12, "name": "Vector Database at Scale",           "tech": "Qdrant, Redis",                        "port": 8012},
    {"id": 13, "name": "Agent Memory System",                "tech": "Redis, Qdrant, Anthropic SDK",         "port": 8013},
    {"id": 14, "name": "Production Inference Server",        "tech": "vLLM, Prometheus, Kubernetes",         "port": 8014},
    {"id": 15, "name": "Human-in-the-Loop Workflow",         "tech": "LangGraph, WebSocket, FastAPI",        "port": 8015},
    {"id": 16, "name": "Agentic Automation Pipeline",        "tech": "Celery, Redis, FastAPI",               "port": 8016},
    {"id": 17, "name": "Legal AI Domain Benchmark",          "tech": "DeepEval, ROUGE, sentence-transformers","port": 8017},
    {"id": 18, "name": "Sunflower Multilingual Translator",  "tech": "Sunflower-14B, vLLM, FastAPI",         "port": 8018},
    {"id": 19, "name": "Uganda Community Health Bot",        "tech": "Sunflower-14B, FastAPI",               "port": 8019},
    {"id": 20, "name": "Model Drift Monitor",                "tech": "sentence-transformers, scipy, SQLite",  "port": 8020},
]


@mcp.tool()
def ask_health_bot(question: str, language: str = "eng") -> dict:
    """Ask the Uganda Community Health Bot a health question.

    Responds in Luganda (lug), Acholi (ach), Ateso (teo), Runyankole (nyn),
    or English (eng). Automatically detects emergencies and returns the Uganda
    health emergency number (0800-100-066) with an urgent referral message.

    Args:
        question: The health question to ask.
        language: ISO 639-3 language code — eng, lug, ach, teo, or nyn.
    """
    return health_bot.ask(language_code=language, question=question)


@mcp.tool()
def route_query(query: str) -> dict:
    """Classify a query by complexity and recommend the optimal Claude model.

    Returns the recommended model (Haiku / Sonnet / Opus), complexity level,
    reasoning, and estimated cost per 1 000 input tokens.

    Args:
        query: The text prompt or question to classify.
    """
    decision = router.classify(query)
    return {
        "recommended_model": decision.model,
        "complexity": decision.complexity,
        "reason": decision.reason,
        "estimated_cost_per_1k_tokens_usd": decision.estimated_cost_per_1k_tokens_usd,
    }


@mcp.tool()
def list_projects(search: str = "") -> list[dict]:
    """List all 20 AI engineering projects in the portfolio.

    Args:
        search: Optional keyword to filter by project name or tech stack.
    """
    if not search:
        return PROJECTS
    term = search.lower()
    return [p for p in PROJECTS if term in p["name"].lower() or term in p["tech"].lower()]


@mcp.resource("projects://all")
def projects_resource() -> str:
    """All 20 AI projects as a markdown summary."""
    lines = ["# AI Engineering Portfolio — 20 Projects\n"]
    for p in PROJECTS:
        lines.append(f"## {p['id']:02d}. {p['name']}")
        lines.append(f"- **Tech**: {p['tech']}")
        lines.append(f"- **Port**: {p['port']}")
        lines.append(f"- **GitHub**: https://github.com/EricPeter/ai-projects/tree/main/{p['id']:02d}-{p['name'].lower().replace(' ', '-').replace(',', '')}")
        lines.append("")
    return "\n".join(lines)


@mcp.resource("health://languages")
def supported_languages() -> str:
    """Languages supported by the Uganda Health Bot."""
    return (
        "# Supported Languages\n\n"
        "| Code | Language   |\n"
        "|------|------------|\n"
        "| eng  | English    |\n"
        "| lug  | Luganda    |\n"
        "| ach  | Acholi     |\n"
        "| teo  | Ateso      |\n"
        "| nyn  | Runyankole |\n"
    )
