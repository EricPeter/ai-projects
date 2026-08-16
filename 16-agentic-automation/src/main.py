import hashlib
import hmac
import json
import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from .idempotency import get_result, is_duplicate, mark_processed
from .tasks import celery, process_job

app = FastAPI(title="Agentic Automation Pipeline")

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def _validate_github_signature(body: bytes, signature: str) -> bool:
    if not GITHUB_WEBHOOK_SECRET:
        return True
    expected = "sha256=" + hmac.new(GITHUB_WEBHOOK_SECRET.encode(), body, "sha256").hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhooks/github")
async def github_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not _validate_github_signature(body, sig):
        raise HTTPException(401, "Invalid signature")

    payload = json.loads(body)
    idempotency_key = request.headers.get("X-GitHub-Delivery", str(uuid.uuid4()))

    if is_duplicate(idempotency_key):
        return {"status": "duplicate", "result": get_result(idempotency_key)}

    job_id = str(uuid.uuid4())
    pr_body = payload.get("pull_request", {}).get("body", "No description")
    pr_title = payload.get("pull_request", {}).get("title", "")
    process_job.delay(job_id, {"type": "summarise_pr", "content": f"{pr_title}\n\n{pr_body}"}, idempotency_key)

    return {"job_id": job_id, "status": "queued"}


class JobRequest(BaseModel):
    type: str = "generic"
    content: str


@app.post("/jobs")
def submit_job(body: JobRequest):
    job_id = str(uuid.uuid4())
    idempotency_key = hashlib.sha256(f"{body.type}:{body.content}".encode()).hexdigest()

    if is_duplicate(idempotency_key):
        return {"status": "duplicate", "result": get_result(idempotency_key)}

    task = process_job.delay(job_id, body.model_dump(), idempotency_key)
    return {"job_id": job_id, "task_id": task.id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    result = celery.AsyncResult(job_id)
    return {"task_id": job_id, "status": result.status, "result": result.result if result.ready() else None}


@app.get("/health")
def health():
    return {"status": "ok"}
