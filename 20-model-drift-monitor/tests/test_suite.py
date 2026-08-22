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


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_probe_runs_and_returns_results(base_url):
    r = httpx.post(f"{base_url}/probe", timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert "probes" in body or "alerts" in body or "results" in body


def test_dashboard_returns_status(base_url):
    r = httpx.get(f"{base_url}/dashboard", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "status" in body or "probes" in body


def test_alerts_endpoint(base_url):
    r = httpx.get(f"{base_url}/alerts", timeout=10)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── Regression ────────────────────────────────────────────────────────────────

def test_health_schema_stable(base_url):
    keys = sorted(httpx.get(f"{base_url}/health", timeout=10).json().keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"]


def test_probe_result_schema_stable(base_url):
    body = httpx.post(f"{base_url}/probe", timeout=30).json()
    keys = sorted(body.keys())
    stored = _baseline("probe_keys", {"keys": keys})
    assert keys == stored["keys"]


# ── A/B ── single probe run vs repeated probe ─────────────────────────────────

def test_probe_idempotency(base_url):
    r1 = httpx.post(f"{base_url}/probe", timeout=30).json()
    r2 = httpx.post(f"{base_url}/probe", timeout=30).json()

    probe_count_1 = len(r1.get("probes", r1.get("results", [])))
    probe_count_2 = len(r2.get("probes", r2.get("results", [])))

    print(f"\n  run 1: {probe_count_1} probe results  run 2: {probe_count_2} probe results")
    assert probe_count_1 == probe_count_2, "probe count should be stable across runs"
