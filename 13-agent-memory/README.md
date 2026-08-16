# Agent Memory System

Multi-layer persistent memory: short-term (Redis), long-term (Qdrant), context compression, and cross-session recall.

## Memory Layers

| Layer | Storage | Capacity | TTL |
|-------|---------|---------|-----|
| Short-term | Redis | Last 10 turns | 1 hour |
| Long-term | Qdrant | 1000 facts/user | 30 days |
| Compressed | In-context | Rolling summary | Session |

## Setup

```bash
cd 13-agent-memory
cp .env.example .env   # add ANTHROPIC_API_KEY
docker-compose up      # starts Redis + Qdrant + API
```

## Usage

```bash
SESSION=sess-001
USER=alice

# Chat (memory is automatic)
curl -X POST http://localhost:8013/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"My name is Alice and I work in AI research.\", \"user_id\": \"$USER\"}"

# Second session — long-term facts recalled from previous
curl -X POST http://localhost:8013/sessions/sess-002/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What do you know about me?\", \"user_id\": \"$USER\"}"

# View all long-term memories for a user
curl http://localhost:8013/users/$USER/memories
```
