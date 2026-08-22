import os
import pytest
from datasets import load_dataset

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8021")

@pytest.fixture(scope="session")
def sst2_sample():
    ds = load_dataset("glue", "sst2", split="validation[:8]")
    return [{"sentence": row["sentence"], "label": row["label"]} for row in ds]
