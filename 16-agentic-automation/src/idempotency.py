import os
import json
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
TTL = 86_400

_r = redis.from_url(REDIS_URL, decode_responses=True)


def is_duplicate(key: str) -> bool:
    return bool(_r.get(f"idempotency:{key}"))


def mark_processed(key: str, result: dict) -> None:
    _r.setex(f"idempotency:{key}", TTL, json.dumps(result))


def get_result(key: str) -> dict | None:
    val = _r.get(f"idempotency:{key}")
    return json.loads(val) if val else None
