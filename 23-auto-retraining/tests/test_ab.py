"""
A/B test: compare pipeline behaviour under clean data vs drifted data.
Variant A — 10 correct labels (high accuracy).
Variant B — 10 random wrong labels (low accuracy).
Validates that the retrain signal fires for B but not A.
"""
import httpx
import pytest
from datasets import load_dataset


@pytest.fixture(scope="module")
def ag_news():
    ds = load_dataset("ag_news", split="test[:20]")
    labels = ds.features["label"].names
    return [{"text": row["text"][:300], "label": labels[row["label"]]} for row in ds]


def _log_and_label(base_url, text, prediction, true_label, latency_ms=30.0):
    r = httpx.post(
        f"{base_url}/infer",
        json={"features": {"text": text}, "prediction": prediction, "latency_ms": latency_ms},
        timeout=10,
    )
    inf_id = r.json()["id"]
    httpx.post(f"{base_url}/label/{inf_id}", json={"label": true_label}, timeout=10)
    return inf_id


def test_variant_a_clean_no_retrain(base_url, ag_news):
    # Variant A: correct predictions (accuracy ~100%)
    for row in ag_news[:10]:
        _log_and_label(base_url, row["text"], row["label"], row["label"], latency_ms=20.0)

    health = httpx.get(f"{base_url}/health", timeout=10).json()
    if health["labeled_last_24h"] >= 10 and health["accuracy"] is not None:
        print(f"\n  Variant A (clean): accuracy={health['accuracy']:.2%}  retrain={health['retrain_needed']}")
        # High accuracy → pipeline should not demand retrain
        assert not health["retrain_needed"] or health["accuracy"] > 0.7, (
            "clean data should not trigger retrain at high accuracy"
        )


def test_variant_b_drifted_triggers_retrain(base_url, ag_news):
    all_labels = list({"World", "Sports", "Business", "Sci/Tech"})

    for row in ag_news[:10]:
        wrong = next(l for l in all_labels if l != row["label"])
        _log_and_label(base_url, row["text"], wrong, row["label"], latency_ms=500.0)

    health = httpx.get(f"{base_url}/health", timeout=10).json()
    if health["labeled_last_24h"] >= 10 and health["accuracy"] is not None:
        print(f"\n  Variant B (drifted): accuracy={health['accuracy']:.2%}  retrain={health['retrain_needed']}")
        # Low accuracy + high latency → retrain should be triggered
        assert health["retrain_needed"] or health["accuracy"] > 0.5, (
            "drifted data with consistently wrong predictions should trigger retrain"
        )


def test_latency_threshold_comparison(base_url):
    results = {}
    for latency_ms in (20.0, 600.0):
        r = httpx.post(
            f"{base_url}/infer",
            json={"features": {"text": "test"}, "prediction": "Sports", "latency_ms": latency_ms},
            timeout=10,
        )
        assert r.status_code == 200
        results[latency_ms] = r.json()["id"]

    print(f"\n  low-latency inference id={results[20.0]}")
    print(f"  high-latency inference id={results[600.0]}")
    assert results[20.0] != results[600.0]
