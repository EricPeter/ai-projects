import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_tenant_isolation(client):
    # Register two tenants
    r1 = await client.post("/auth/register", json={"email": "a@acme.com", "password": "pass1", "company_name": "Acme"})
    r2 = await client.post("/auth/register", json={"email": "b@beta.com", "password": "pass2", "company_name": "Beta"})
    assert r1.status_code == 200
    assert r2.status_code == 200

    token1 = r1.json()["token"]
    token2 = r2.json()["token"]
    tenant1_id = r1.json()["tenant_id"]
    tenant2_id = r2.json()["tenant_id"]

    # Each tenant gets a different tenant_id
    assert tenant1_id != tenant2_id

    # Usage for tenant1 doesn't include tenant2's data
    usage1 = await client.get("/usage", headers={"Authorization": f"Bearer {token1}"})
    usage2 = await client.get("/usage", headers={"Authorization": f"Bearer {token2}"})
    assert usage1.json()["tenant_id"] == tenant1_id
    assert usage2.json()["tenant_id"] == tenant2_id
    assert usage1.json()["tenant_id"] != usage2.json()["tenant_id"]
