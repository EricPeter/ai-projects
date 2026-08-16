import asyncio
import os
import time
from dataclasses import dataclass, field

import httpx

INSTANCES = [
    os.getenv("VLLM_INSTANCE_1", "http://localhost:8080"),
    os.getenv("VLLM_INSTANCE_2", "http://localhost:8081"),
]


@dataclass
class Instance:
    url: str
    healthy: bool = True
    queue_depth: int = 0
    last_check: float = field(default_factory=time.time)


_instances = [Instance(url=u) for u in INSTANCES]


async def get_best_instance() -> str:
    healthy = [i for i in _instances if i.healthy]
    if not healthy:
        raise RuntimeError("No healthy vLLM instances available")
    return min(healthy, key=lambda i: i.queue_depth).url


async def health_check_loop():
    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            for inst in _instances:
                try:
                    r = await client.get(f"{inst.url}/health")
                    inst.healthy = r.status_code == 200
                except Exception:
                    inst.healthy = False
                inst.last_check = time.time()
            await asyncio.sleep(10)


def get_instances_status() -> list[dict]:
    return [{"url": i.url, "healthy": i.healthy, "queue_depth": i.queue_depth} for i in _instances]
