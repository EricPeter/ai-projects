import io
import sys
import httpx
from reportlab.pdfgen import canvas

API = "http://localhost:8000"


def make_sample_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "Page 1: Introduction to RAG Systems")
    c.drawString(72, 720, "Retrieval Augmented Generation combines search with LLMs.")
    c.showPage()
    c.drawString(72, 750, "Page 2: Hybrid Search")
    c.drawString(72, 720, "Hybrid search combines dense vector search with BM25 keyword search.")
    c.save()
    return buf.getvalue()


def main():
    client = httpx.Client(base_url=API, timeout=60)

    print("1. Uploading sample PDF...")
    pdf_bytes = make_sample_pdf()
    resp = client.post("/upload", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})
    resp.raise_for_status()
    print("   Upload response:", resp.json())

    print("2. Running query...")
    resp = client.post("/query", json={"question": "What is hybrid search?"})
    resp.raise_for_status()
    data = resp.json()
    print("   Answer:", data["answer"][:200])
    print("   Citations (pages):", data["citations"])
    print("   Sources:", data["sources"])


if __name__ == "__main__":
    main()
