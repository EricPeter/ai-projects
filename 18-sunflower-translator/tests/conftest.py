import os
import pytest
from datasets import load_dataset

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8080")

@pytest.fixture(scope="session")
def flores_sample():
    try:
        ds = load_dataset("facebook/flores", "eng_Latn-lug_Latn", split="devtest[:5]", trust_remote_code=True)
        return [{"text": r["sentence_eng_Latn"], "lang": "eng"} for r in ds]
    except Exception:
        return [
            {"text": "Good morning, how are you?", "lang": "eng"},
            {"text": "The hospital is nearby.", "lang": "eng"},
            {"text": "Clean water is essential.", "lang": "eng"},
        ]
