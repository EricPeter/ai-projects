import httpx

API = "http://localhost:8007"
SAMPLE = """
ACME Widget Corp Technical Docs

Our flagship product, the Widget Pro 3000, uses a patented dual-core turbine mechanism.
It operates at temperatures between -10°C and 85°C and has a throughput of 500 units/hour.
The maintenance cycle is every 6 months or 10,000 operating hours, whichever comes first.
Contact support@acme.example.com for warranty claims.
"""

QUERIES = [
    "What is the operating temperature range of the Widget Pro 3000?",
    "How often should maintenance be performed?",
    "What is the throughput of the product?",
]


def main():
    client = httpx.Client(base_url=API, timeout=120)

    print("1. Ingesting sample document...")
    resp = client.post("/upload", files={"file": ("docs.txt", SAMPLE.encode(), "text/plain")})
    resp.raise_for_status()
    print("  ", resp.json())

    for i, q in enumerate(QUERIES, 1):
        print(f"\n{i+1}. Query: {q}")
        resp = client.post("/query", json={"question": q})
        resp.raise_for_status()
        data = resp.json()
        print("   Answer:", data["answer"][:300])


if __name__ == "__main__":
    main()
