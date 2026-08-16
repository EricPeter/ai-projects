import hashlib
import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/rag.db")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT UNIQUE NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[384]
        );
        CREATE TABLE IF NOT EXISTS bm25_index (
            doc_id TEXT NOT NULL,
            index_data BLOB NOT NULL
        );
    """)
    conn.commit()


def _chunk_text(text: str, page_number: int) -> list[dict[str, Any]]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append({"content": chunk_text, "page_number": page_number})
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_pdf(pdf_bytes: bytes, doc_id: str) -> int:
    reader = PdfReader(pdf_bytes if hasattr(pdf_bytes, "read") else __import__("io").BytesIO(pdf_bytes))
    all_chunks: list[dict[str, Any]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        all_chunks.extend(_chunk_text(text, page_num))

    if not all_chunks:
        return 0

    model = get_embed_model()
    texts = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    conn = get_db()
    inserted = 0
    chunk_ids = []
    for chunk, embedding in zip(all_chunks, embeddings):
        content_hash = hashlib.sha256(chunk["content"].encode()).hexdigest()
        try:
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, page_number, content, content_hash) VALUES (?, ?, ?, ?)",
                (doc_id, chunk["page_number"], chunk["content"], content_hash),
            )
            chunk_id = cur.lastrowid
            vec_bytes = struct.pack(f"{len(embedding)}f", *embedding.tolist())
            conn.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, vec_bytes),
            )
            chunk_ids.append(chunk_id)
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    # Update BM25 index
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    conn.execute("DELETE FROM bm25_index WHERE doc_id = ?", (doc_id,))
    conn.execute(
        "INSERT INTO bm25_index (doc_id, index_data) VALUES (?, ?)",
        (doc_id, json.dumps({"corpus": texts, "pages": [c["page_number"] for c in all_chunks], "ids": chunk_ids})),
    )

    conn.commit()
    conn.close()
    return inserted
