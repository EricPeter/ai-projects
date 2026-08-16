import os
import httpx

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080")
MODEL = os.getenv("MODEL_NAME", "Sunbird/Sunflower-14B-W8A8")


def _build_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def complete(system: str, user: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": _build_messages(system, user),
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def stream(system: str, user: str, max_tokens: int = 512, temperature: float = 0.3):
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": _build_messages(system, user),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    import json
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
