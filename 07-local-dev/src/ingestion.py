import os
import uuid
from pathlib import Path

import lancedb
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/lancedb")
TABLE_NAME = "documents"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 300

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_table():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(DB_PATH))
    try:
        return db.open_table(TABLE_NAME)
    except Exception:
        import pyarrow as pa
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("source", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
        ])
        return db.create_table(TABLE_NAME, schema=schema)


def _chunk(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[i: i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]


def ingest_text(text: str, source: str = "unknown") -> int:
    model = get_model()
    chunks = [c for c in _chunk(text) if c.strip()]
    if not chunks:
        return 0

    embeddings = model.encode(chunks, normalize_embeddings=True)
    table = _get_table()
    records = [
        {"id": str(uuid.uuid4()), "content": c, "source": source, "chunk_index": i, "vector": e.tolist()}
        for i, (c, e) in enumerate(zip(chunks, embeddings))
    ]
    table.add(records)
    return len(records)


def ingest_pdf(pdf_bytes: bytes, filename: str) -> int:
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return ingest_text(full_text, source=filename)
