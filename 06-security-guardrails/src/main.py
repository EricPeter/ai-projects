import time
from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import anthropic

from .guardrails.injection_detector import detect as detect_injection
from .guardrails.pii_redactor import redact
from .guardrails.output_filter import filter_output
from .audit_logger import log_blocked

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Security Guardrail Middleware")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


class ChatRequest(BaseModel):
    message: str
    system: str = "You are a helpful assistant."


@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest):
    start = time.perf_counter()
    client_ip = get_remote_address(request)
    triggered = []

    # 1. Injection detection
    injection = detect_injection(body.message)
    if injection.is_injection:
        log_blocked("prompt_injection", client_ip, body.message)
        raise HTTPException(400, detail={"error": "Prompt injection detected", "confidence": injection.confidence})

    # 2. PII redaction on input
    redaction = redact(body.message)
    if redaction.pii_detected:
        triggered.append("pii_redacted")

    # 3. Call LLM with redacted input
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=body.system,
        messages=[{"role": "user", "content": redaction.redacted_text}],
    )
    raw_response = msg.content[0].text

    # 4. Output filter
    filtered = filter_output(raw_response)
    if not filtered.is_safe:
        triggered.extend(filtered.issues)
        log_blocked(str(filtered.issues), client_ip, body.message)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    return {
        "response": filtered.filtered_text or raw_response,
        "guardrails_triggered": triggered,
        "pii_detected": redaction.pii_detected,
        "processing_time_ms": elapsed_ms,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
