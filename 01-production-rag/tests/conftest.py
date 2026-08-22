import os
import pytest


@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def squad_sample():
    try:
        from datasets import load_dataset
        ds = load_dataset("squad", split="validation[:5]")
        return [{"question": r["question"], "context": r["context"][:300]} for r in ds]
    except Exception:
        return [
            {"question": "What is the capital of France?", "context": "Paris is the capital of France."},
            {"question": "Who invented the telephone?", "context": "Alexander Graham Bell invented the telephone in 1876."},
            {"question": "What is machine learning?", "context": "Machine learning is a subset of artificial intelligence that learns from data."},
        ]
