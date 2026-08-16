import time
import random
import uuid
import httpx
import numpy as np

API = "http://localhost:8012"
N = 1_000  # reduced for demo; raise to 10k for real benchmark
COLLECTION = "benchmark"
DIM = 384


def main():
    client = httpx.Client(base_url=API, timeout=120)

    print(f"Creating collection '{COLLECTION}'...")
    client.post("/collections", json={"name": COLLECTION, "vector_size": DIM}).raise_for_status()

    print(f"Inserting {N} documents...")
    batch_size = 100
    for i in range(0, N, batch_size):
        docs = [{"text": f"Sample document {i+j} about topic {random.randint(1, 20)}", "topic": random.randint(1, 20)} for j in range(batch_size)]
        client.post("/ingest", json={"collection": COLLECTION, "documents": docs}).raise_for_status()
    print("Done inserting.")

    print("\nBenchmark (QPS):")
    print(f"{'k':>5} | {'QPS':>10} | {'Latency ms':>12}")
    print("-" * 35)
    for k in [1, 5, 10, 50]:
        times = []
        for _ in range(20):
            query = f"document about topic {random.randint(1, 20)}"
            start = time.perf_counter()
            client.post("/search", json={"collection": COLLECTION, "query": query, "k": k}).raise_for_status()
            times.append(time.perf_counter() - start)
        avg_ms = sum(times) / len(times) * 1000
        qps = 1000 / avg_ms
        print(f"{k:>5} | {qps:>10.1f} | {avg_ms:>10.1f}ms")


if __name__ == "__main__":
    main()
