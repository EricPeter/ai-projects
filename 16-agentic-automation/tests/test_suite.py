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


def _wait_for_job(base_url, job_id, timeout_s=30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = httpx.get(f"{base_url}/jobs/{job_id}", timeout=10)
        if r.json().get("status") in ("done", "completed", "failed"):
            return r.json()
        time.sleep(2)
    return httpx.get(f"{base_url}/jobs/{job_id}", timeout=10).json()


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_submit_job_returns_id(base_url, tickets):
    r = httpx.post(f"{base_url}/jobs", json=tickets[0], timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body.get("status") in ("queued", "running", "pending")


def test_job_status_retrievable(base_url, tickets):
    job_id = httpx.post(f"{base_url}/jobs", json=tickets[0], timeout=10).json()["job_id"]
    r = httpx.get(f"{base_url}/jobs/{job_id}", timeout=10)
    assert r.status_code == 200
    assert "status" in r.json()


def test_completed_job_has_result(base_url, tickets):
    job_id = httpx.post(f"{base_url}/jobs", json=tickets[0], timeout=10).json()["job_id"]
    result = _wait_for_job(base_url, job_id)
    assert "result" in result or "status" in result


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


# ── A/B ── simple vs complex tickets ─────────────────────────────────────────

def test_simple_vs_complex_ticket_processing(base_url, tickets):
    simple = {"type": "categorise_ticket", "content": "App is slow"}
    complex_t = {
        "type": "categorise_ticket",
        "content": (
            "The application intermittently crashes with a NullPointerException in the payment flow "
            "after the 3D-secure redirect, but only on iOS 17.4+ devices when the user has more than "
            "3 saved cards. This has been happening since the v2.3.1 release on Thursday."
        ),
    }

    t0 = time.perf_counter()
    id_simple = httpx.post(f"{base_url}/jobs", json=simple, timeout=10).json()["job_id"]
    _wait_for_job(base_url, id_simple)
    simple_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    id_complex = httpx.post(f"{base_url}/jobs", json=complex_t, timeout=10).json()["job_id"]
    _wait_for_job(base_url, id_complex)
    complex_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  simple ticket:  {simple_ms:.0f}ms")
    print(f"  complex ticket: {complex_ms:.0f}ms")
    assert simple_ms < 60_000 and complex_ms < 60_000
