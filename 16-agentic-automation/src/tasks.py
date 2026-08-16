import os
from celery import Celery
import anthropic

from .idempotency import mark_processed

BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery = Celery("agentic", broker=BROKER, backend=BACKEND)
celery.conf.task_serializer = "json"
celery.conf.result_expires = 3600

client = anthropic.Anthropic()


@celery.task(bind=True, max_retries=3, default_retry_delay=30)
def process_job(self, job_id: str, payload: dict, idempotency_key: str):
    try:
        job_type = payload.get("type", "generic")
        content = payload.get("content", "")

        prompts = {
            "summarise_pr": f"Summarise this pull request in 3 bullet points:\n\n{content}",
            "categorise_ticket": f"Categorise this support ticket (Bug/Feature/Question/Other) and provide a 1-sentence summary:\n\n{content}",
            "generic": f"Process this automation task:\n\n{content}",
        }
        prompt = prompts.get(job_type, prompts["generic"])

        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        result = {"job_id": job_id, "output": msg.content[0].text, "status": "completed"}
        mark_processed(idempotency_key, result)
        return result

    except Exception as exc:
        countdown = 2 ** self.request.retries * 30
        raise self.retry(exc=exc, countdown=countdown)
