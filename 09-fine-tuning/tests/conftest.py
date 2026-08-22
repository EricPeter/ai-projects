import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def lora_dataset():
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train[:5]")
    return [{"instruction": r["instruction"], "output": r["output"]} for r in ds]
