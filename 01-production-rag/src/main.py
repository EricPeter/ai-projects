import io
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .ingestion import ingest_pdf
from .rag_graph import rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models on startup
    from .ingestion import get_embed_model
    from .retrieval import get_cross_encoder
    get_embed_model()
    get_cross_encoder()
    yield


app = FastAPI(title="Production RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    sources: list[dict]


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    contents = await file.read()
    doc_id = str(uuid.uuid4())
    count = ingest_pdf(contents, doc_id)

    return {"doc_id": doc_id, "chunks_indexed": count, "filename": file.filename}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    state = rag_graph.invoke({"question": request.question, "documents": [], "answer": "", "citations": []})

    return QueryResponse(
        answer=state["answer"],
        citations=state["citations"],
        sources=[
            {"page": d["page_number"], "content": d["content"][:200] + "...", "score": round(d.get("rerank_score", 0), 4)}
            for d in state["documents"]
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
