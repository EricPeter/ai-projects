import os
import pytest

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8015")

@pytest.fixture(scope="session")
def tasks():
    return [
        "Archive all log files older than 30 days",
        "Delete all inactive user accounts older than 2 years",
        "Send promotional email to all subscribers",
    ]
