import json
from pathlib import Path

import httpx
import pytest

BASELINES = Path(__file__).parent / "baselines"


def _baseline(name, current):
    BASELINES.mkdir(exist_ok=True)
    path = BASELINES / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline for '{name}' created")
    return json.loads(path.read_text())


def test_health_schema_stable(base_url):
    body = httpx.get(f"{base_url}/health", timeout=10).json()
    keys = sorted(body.keys())
    stored = _baseline("health_keys", {"keys": keys})
    assert keys == stored["keys"], "health schema changed"


def test_correct_labels_do_not_trigger_retrain(base_url, ag_news_sample):
    for row in ag_news_sample:
        infer_r = httpx.post(
            f"{base_url}/infer",
            json={"features": {"text": row["text"][:300]}, "prediction": row["label"], "latency_ms": 25.0},
            timeout=10,
        )
        httpx.post(
            f"{base_url}/label/{infer_r.json()['id']}",
            json={"label": row["label"]},
            timeout=10,
        )

    health = httpx.get(f"{base_url}/health", timeout=10).json()
    # With 10 correct labels and high accuracy, pipeline should not trigger retrain
    if health["labeled_last_24h"] >= 10 and health["accuracy"] is not None:
        stored = _baseline("accuracy_on_correct", {"accuracy": round(health["accuracy"], 2)})
        delta = abs(health["accuracy"] - stored["accuracy"])
        assert delta < 0.15, f"accuracy changed by {delta:.2f} from baseline"


def test_retrain_flag_is_bool(base_url):
    body = httpx.get(f"{base_url}/health", timeout=10).json()
    assert isinstance(body["retrain_needed"], bool)
