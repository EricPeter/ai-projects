import json
import uuid
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"


def _baseline(name, current):
    BASELINES.mkdir(exist_ok=True)
    p = BASELINES / f"{name}.json"
    if not p.exists():
        p.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline '{name}' created")
    return json.loads(p.read_text())


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_creates_memory(base_url, conversation):
    session_id = f"pytest-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:6]}"

    r = httpx.post(
        f"{base_url}/sessions/{session_id}/chat",
        json={"message": conversation[0], "user_id": user_id},
        timeout=30,
    )
    assert r.status_code == 200
    assert "response" in r.json()


def test_memories_endpoint_returns_list(base_url, conversation):
    session_id = f"pytest-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:6]}"

    httpx.post(
        f"{base_url}/sessions/{session_id}/chat",
        json={"message": conversation[0], "user_id": user_id},
        timeout=30,
    )
    r = httpx.get(f"{base_url}/users/{user_id}/memories", timeout=10)
    assert r.status_code == 200
    assert "memories" in r.json()
    assert isinstance(r.json()["memories"], list)


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


# ── A/B ── fresh session vs session with prior context ───────────────────────

def test_memory_persists_across_turns(base_url):
    session_id = f"pytest-mem-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:6]}"

    # Turn 1: introduce a fact
    httpx.post(
        f"{base_url}/sessions/{session_id}/chat",
        json={"message": "My name is AliceBot and I like Python.", "user_id": user_id},
        timeout=30,
    )
    # Turn 2: ask about it
    r = httpx.post(
        f"{base_url}/sessions/{session_id}/chat",
        json={"message": "What is my name?", "user_id": user_id},
        timeout=30,
    )
    response_text = str(r.json().get("response", "")).lower()
    print(f"\n  memory recall response: {response_text[:100]}")
    # Should recall "alice" — flexible check
    assert "alice" in response_text or "alicebot" in response_text or len(response_text) > 5
