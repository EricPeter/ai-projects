# Multi-Tenant SaaS Agent

Production-ready multi-tenant AI agent with JWT auth, per-tenant rate limiting, Stripe billing, and data isolation.

## Features
- **Tenant isolation** — every DB query is scoped to `tenant_id`
- **JWT authentication** — register, login, bearer token
- **Rate limiting** — free: 10 req/min, pro: 100 req/min
- **Usage tracking** — cost per tenant, input/output tokens
- **Audit log** — every chat action is logged

## Setup

```bash
cd 10-multi-tenant-saas
cp .env.example .env
uv sync
uv run uvicorn src.main:app --reload
```

## Usage

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@acme.com", "password": "secret", "company_name": "Acme"}'

# Chat (use token from register response)
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Check usage
curl http://localhost:8000/usage \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Tests

```bash
uv run pytest tests/ -v
```
