import os
import pytest

@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8010")

@pytest.fixture(scope="session")
def tenant_a():
    return {"email": "alice@acme-test.com", "password": "testpass123", "company_name": "AcmeTest"}

@pytest.fixture(scope="session")
def tenant_b():
    return {"email": "bob@globaltest.com", "password": "testpass456", "company_name": "GlobalTest"}
