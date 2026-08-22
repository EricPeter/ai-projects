import os
import pytest
from datasets import load_dataset

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8023")

@pytest.fixture(scope="session")
def ag_news_sample():
    ds = load_dataset("ag_news", split="test[:10]")
    return [{"text": row["text"], "label": ds.features["label"].names[row["label"]]} for row in ds]
