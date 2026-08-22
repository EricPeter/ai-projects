import os
import pytest
from datasets import load_dataset

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8012")

@pytest.fixture(scope="session")
def msmarco_sample():
    ds = load_dataset("ms_marco", "v1.1", split="train[:5]")
    return [
        {"text": row["passages"]["passage_text"][0][:300], "query": row["query"]}
        for row in ds
    ]
