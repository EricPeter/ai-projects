import os
import pytest

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8013")

@pytest.fixture(scope="session")
def conversation():
    return [
        "My name is TestUser and I work in AI research.",
        "I prefer concise answers.",
        "What do you know about me?",
    ]
