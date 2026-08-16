import json
import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_TURNS = 10
SESSION_TTL = 3600  # 1 hour

_redis: redis.Redis | None = None


def _r() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _key(session_id: str) -> str:
    return f"session:{session_id}:turns"


def add_turn(session_id: str, role: str, content: str) -> None:
    r = _r()
    key = _key(session_id)
    turn = json.dumps({"role": role, "content": content})
    r.rpush(key, turn)
    r.ltrim(key, -MAX_TURNS, -1)  # keep last N turns
    r.expire(key, SESSION_TTL)


def get_turns(session_id: str) -> list[dict]:
    r = _r()
    raw = r.lrange(_key(session_id), 0, -1)
    return [json.loads(t) for t in raw]


def clear(session_id: str) -> None:
    _r().delete(_key(session_id))
