from dataclasses import dataclass

import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class RetrievedChunk:
    content: str
    page: int
    score: float


class SimpleRAG:
    def __init__(self):
        self._chunks: list[str] = []
        self._pages: list[int] = []
        self._embeddings: np.ndarray | None = None
        self._client = anthropic.Anthropic()

    def index(self, text: str, page: int = 1) -> None:
        # Split into ~200-word chunks
        words = text.split()
        for i in range(0, len(words), 200):
            chunk = " ".join(words[i: i + 200])
            if chunk.strip():
                self._chunks.append(chunk)
                self._pages.append(page)
        self._embeddings = _model.encode(self._chunks, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if self._embeddings is None:
            return []
        q_emb = _model.encode(query, normalize_embeddings=True)
        scores = (self._embeddings @ q_emb).tolist()
        top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [RetrievedChunk(self._chunks[i], self._pages[i], s) for i, s in top]

    def answer(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.retrieve(question)
        context = "\n\n".join(f"[Page {c.page}] {c.content}" for c in chunks)
        msg = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system="Answer using only the provided context. Be concise and precise.",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return msg.content[0].text, chunks
