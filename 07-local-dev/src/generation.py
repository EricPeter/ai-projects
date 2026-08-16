import os
import ollama


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def generate(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['content']}" for c in context_chunks)
    prompt = f"Answer using only the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content
