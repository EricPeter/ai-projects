import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).parent.parent
