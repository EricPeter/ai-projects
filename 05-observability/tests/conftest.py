import os
import pytest


@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8005")


@pytest.fixture(scope="session")
def chat_inputs():
    try:
        from datasets import load_dataset
        ds = load_dataset("squad", split="validation[:5]")
        return [r["question"] for r in ds]
    except Exception:
        return [
            "Explain quantum entanglement simply.",
            "What is the difference between TCP and UDP?",
            "How does a neural network learn?",
        ]
