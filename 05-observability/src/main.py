import time
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import anthropic

from .telemetry import setup_telemetry, get_tracer
from .cost_calculator import calculate_cost

app = FastAPI(title="AI Observability Proxy")

# Prometheus metrics
requests_total = Counter("ai_requests_total", "Total requests", ["model", "status"])
cost_total = Counter("ai_cost_dollars_total", "Total cost USD", ["model"])
latency_hist = Histogram(
    "ai_latency_seconds", "Request latency", ["model"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60],
)
errors_total = Counter("ai_errors_total", "Total errors", ["model", "error_type"])

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"


class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful assistant."


@app.on_event("startup")
def startup():
    setup_telemetry()


@app.post("/chat")
def chat(request: ChatRequest):
    tracer = get_tracer()
    start = time.perf_counter()

    with tracer.start_as_current_span("chat_request") as span:
        span.set_attribute("model", MODEL)
        span.set_attribute("input_length", len(request.message))

        try:
            with tracer.start_as_current_span("llm_call"):
                msg = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=request.system,
                    messages=[{"role": "user", "content": request.message}],
                )

            elapsed = time.perf_counter() - start
            cost = calculate_cost(MODEL, msg.usage.input_tokens, msg.usage.output_tokens)

            span.set_attribute("input_tokens", msg.usage.input_tokens)
            span.set_attribute("output_tokens", msg.usage.output_tokens)
            span.set_attribute("cost_usd", cost)

            requests_total.labels(model=MODEL, status="success").inc()
            cost_total.labels(model=MODEL).inc(cost)
            latency_hist.labels(model=MODEL).observe(elapsed)

            return {
                "response": msg.content[0].text,
                "model": MODEL,
                "cost_usd": round(cost, 8),
                "latency_ms": round(elapsed * 1000, 1),
                "tokens": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens},
            }

        except Exception as e:
            errors_total.labels(model=MODEL, error_type=type(e).__name__).inc()
            requests_total.labels(model=MODEL, status="error").inc()
            raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}
