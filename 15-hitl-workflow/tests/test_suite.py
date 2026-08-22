import json
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


def test_create_workflow_returns_id_and_pending_status(base_url, tasks):
    r = httpx.post(f"{base_url}/workflows", json={"task": tasks[0]}, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "workflow_id" in body
    assert body["status"] == "pending"


def test_pending_workflows_list(base_url, tasks):
    httpx.post(f"{base_url}/workflows", json={"task": tasks[1]}, timeout=10)
    r = httpx.get(f"{base_url}/workflows/pending", timeout=10)
    assert r.status_code == 200
    pending = r.json()
    assert isinstance(pending, list)


def test_approve_workflow(base_url, tasks):
    wf_id = httpx.post(f"{base_url}/workflows", json={"task": tasks[0]}, timeout=10).json()["workflow_id"]
    r = httpx.post(
        f"{base_url}/workflows/{wf_id}/approve",
        json={"decided_by": "test_admin"},
        timeout=10,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "approved"


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_workflow_response_schema_stable(base_url, tasks):
    body = httpx.post(f"{base_url}/workflows", json={"task": tasks[0]}, timeout=10).json()
    keys = sorted(body.keys())
    stored = _baseline("workflow_keys", {"keys": keys})
    assert keys == stored["keys"]


# ── A/B ── high-risk vs low-risk tasks ───────────────────────────────────────

def test_task_risk_classification(base_url, tasks):
    low_risk = "Archive logs older than 30 days"
    high_risk = "Delete all user accounts permanently"

    r_low = httpx.post(f"{base_url}/workflows", json={"task": low_risk}, timeout=10).json()
    r_high = httpx.post(f"{base_url}/workflows", json={"task": high_risk}, timeout=10).json()

    print(f"\n  low-risk task:  status={r_low.get('status')}  risk={r_low.get('risk_level', 'N/A')}")
    print(f"  high-risk task: status={r_high.get('status')}  risk={r_high.get('risk_level', 'N/A')}")

    assert r_low["status"] in ("pending", "approved", "auto-approved")
    assert r_high["status"] in ("pending", "requires_approval")
