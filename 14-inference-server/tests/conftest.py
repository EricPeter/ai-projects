import os
import pytest


@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8014")


@pytest.fixture(scope="session")
def squad_questions():
    try:
        from datasets import load_dataset
        ds = load_dataset("squad", split="validation[:5]")
        return [r["question"] for r in ds]
    except Exception:
        return [
            "What is the capital of France?",
            "Who invented the telephone?",
            "What is machine learning?",
        ]
