import os
import pytest

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8019")

@pytest.fixture(scope="session")
def health_questions():
    return [
        {"question": "What are symptoms of malaria?", "language": "eng"},
        {"question": "Nkwata ddagala ya malaria otya?", "language": "lug"},
        {"question": "How do I prevent cholera?", "language": "eng"},
    ]
