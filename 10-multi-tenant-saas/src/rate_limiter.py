from fastapi import HTTPException, status

# Simple in-memory rate limiter (use Redis in production)
from collections import defaultdict
from datetime import datetime, timezone

_request_counts: dict[str, list[datetime]] = defaultdict(list)

LIMITS = {
    "free": {"requests_per_minute": 10, "tokens_per_day": 1000},
    "pro":  {"requests_per_minute": 100, "tokens_per_day": 100_000},
}


def check_rate_limit(tenant_id: str, plan: str) -> None:
    now = datetime.now(timezone.utc)
    limit = LIMITS.get(plan, LIMITS["free"])

    # Sliding window: keep last 60 seconds
    window = [t for t in _request_counts[tenant_id] if (now - t).seconds < 60]
    _request_counts[tenant_id] = window

    if len(window) >= limit["requests_per_minute"]:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit['requests_per_minute']} req/min on {plan} plan",
            headers={"Retry-After": "60"},
        )

    _request_counts[tenant_id].append(now)
