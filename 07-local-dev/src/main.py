import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import ollama

from .ingestion import ingest_pdf, ingest_text
from .retrieval import search
from .generation import generate, OLLAMA_HOST, OLLAMA_MODEL

app = FastAPI(title="Local-First RAG")


class QueryRequest(BaseModel):
    question: str
    k: int = 5


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename and file.filename.endswith(".pdf"):
        count = ingest_pdf(content, file.filename)
    else:
        count = ingest_text(content.decode("utf-8", errors="ignore"), file.filename or "upload")
    return {"filename": file.filename, "chunks_indexed": count}


@app.post("/query")
def query(request: QueryRequest):
    chunks = search(request.question, k=request.k)
    if not chunks:
        raise HTTPException(404, "No documents indexed yet. Upload a file first.")
    answer = generate(request.question, chunks)
    return {"answer": answer, "sources": chunks}


@app.get("/models")
def list_models():
    client = ollama.Client(host=OLLAMA_HOST)
    models = client.list()
    return {"models": [m.model for m in models.models]}


@app.get("/health")
def health():
    return {"status": "ok", "ollama_host": OLLAMA_HOST, "model": OLLAMA_MODEL}
