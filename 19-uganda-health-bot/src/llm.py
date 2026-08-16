import os
import httpx

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080")
MODEL = os.getenv("MODEL_NAME", "Sunbird/Sunflower-14B-W8A8")


def ask(system: str, user: str, max_tokens: int = 600, temperature: float = 0.3) -> str:
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
