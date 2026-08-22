import json
import time
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


def _register_and_token(base_url, user):
    r = httpx.post(f"{base_url}/auth/register", json=user, timeout=10)
    if r.status_code == 409:
        r = httpx.post(f"{base_url}/auth/login", json=user, timeout=10)
    return r.json().get("token", "")


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_register_returns_token(base_url, tenant_a):
    token = _register_and_token(base_url, tenant_a)
    assert token != "", "registration should return a token"


def test_chat_with_valid_token(base_url, tenant_a):
    token = _register_and_token(base_url, tenant_a)
    r = httpx.post(
        f"{base_url}/chat",
        json={"message": "Hello"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    assert r.status_code == 200


def test_chat_without_token_rejected(base_url):
    r = httpx.post(f"{base_url}/chat", json={"message": "Hello"}, timeout=10)
    assert r.status_code in (401, 403)


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


# ── A/B ── tenant isolation ───────────────────────────────────────────────────

def test_tenant_isolation(base_url, tenant_a, tenant_b):
    token_a = _register_and_token(base_url, tenant_a)
    token_b = _register_and_token(base_url, tenant_b)

    httpx.post(
        f"{base_url}/chat",
        json={"message": "My company secret is XYZ-42"},
        headers={"Authorization": f"Bearer {token_a}"},
        timeout=30,
    )

    r_b = httpx.post(
        f"{base_url}/chat",
        json={"message": "What secret did the other company share?"},
        headers={"Authorization": f"Bearer {token_b}"},
        timeout=30,
    )
    assert r_b.status_code == 200
    response_text = str(r_b.json().get("response", "")).lower()
    assert "xyz-42" not in response_text, "tenant B must not see tenant A's data"
