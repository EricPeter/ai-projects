import os
import pytest

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8016")

@pytest.fixture(scope="session")
def tickets():
    return [
        {"type": "categorise_ticket", "content": "App crashes when I click the submit button"},
        {"type": "categorise_ticket", "content": "Cannot log in after password reset"},
        {"type": "categorise_ticket", "content": "Feature request: dark mode"},
    ]
