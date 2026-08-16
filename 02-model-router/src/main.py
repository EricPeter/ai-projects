import time
import uuid
from fastapi import FastAPI, Response
from pydantic import BaseModel
import anthropic

from .router import classify, estimate_cost
from .cost_tracker import record, summary
from .metrics import requests_total, cost_total, latency, get_metrics_output

app = FastAPI(title="Cost-Optimized Model Router")
client = anthropic.Anthropic()


class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    response: str
    model_used: str
    cost_usd: float
    tokens_used: dict
    routing_reason: str
    complexity: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    decision = classify(request.message)
    request_id = str(uuid.uuid4())

    start = time.perf_counter()
    msg = client.messages.create(
        model=decision.model,
        max_tokens=1024,
        system=request.system,
        messages=[{"role": "user", "content": request.message}],
    )
    elapsed = time.perf_counter() - start

    input_tokens = msg.usage.input_tokens
    output_tokens = msg.usage.output_tokens
    cost = estimate_cost(decision.model, input_tokens, output_tokens)

    requests_total.labels(model=decision.model, complexity=decision.complexity).inc()
    cost_total.labels(model=decision.model).inc(cost)
    latency.labels(model=decision.model).observe(elapsed)

    record(request_id, decision.model, input_tokens, output_tokens, cost, decision.complexity)

    return ChatResponse(
        response=msg.content[0].text,
        model_used=decision.model,
        cost_usd=round(cost, 8),
        tokens_used={"input": input_tokens, "output": output_tokens},
        routing_reason=decision.reason,
        complexity=decision.complexity,
    )


@app.get("/costs")
def get_costs():
    return {"breakdown_by_model": summary()}


@app.get("/metrics")
def metrics():
    data, content_type = get_metrics_output()
    return Response(content=data, media_type=content_type)


@app.get("/health")
def health():
    return {"status": "ok"}
