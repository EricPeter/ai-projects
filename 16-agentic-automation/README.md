# Agentic Automation Pipeline

Async AI job queue with idempotency, retry with exponential backoff, dead-letter handling, and Celery monitoring.

## Features
- **Webhooks** — GitHub PR events trigger AI summarisation automatically
- **Idempotency** — duplicate webhook deliveries are deduplicated via Redis
- **Retry** — exponential backoff (30s, 60s, 120s) on transient failures
- **Flower UI** — visual Celery task monitoring at http://localhost:5555

## Setup

```bash
cd 16-agentic-automation
cp .env.example .env   # add ANTHROPIC_API_KEY
docker-compose up
```

Services:
| Service | URL |
|---------|-----|
| API | http://localhost:8016 |
| Flower | http://localhost:5555 |

## Usage

```bash
# Submit a job directly
curl -X POST http://localhost:8016/jobs \
  -H "Content-Type: application/json" \
  -d '{"type": "categorise_ticket", "content": "My app crashes when I click the submit button"}'

# Check job status
curl http://localhost:8016/jobs/TASK_ID

# Send test GitHub webhook
uv run python scripts/send_test_webhook.py
```

## Job Types

| type | Description |
|------|------------|
| `summarise_pr` | Summarise a pull request |
| `categorise_ticket` | Categorise a support ticket |
| `generic` | Process any automation task |
