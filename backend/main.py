from pathlib import Path

import requests
from config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import rag

DOCS_PATH = Path(settings.DOCS_DIRECTORY)


@app.get("/")
def root():
    return {"message": "RAG API running in Docker", "model": settings.MODEL_NAME}


@app.get("/health")
def health():
    ollama_ok = False
    try:
        r = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception as exc:
        print(f"Ollama health check failed for {settings.OLLAMA_URL!r}: {exc}")
    return {
        "status": "healthy",
        "ollama": "connected" if ollama_ok else "unavailable",
        "ollama_url": settings.OLLAMA_URL,
        "documents": rag.collection.count(),
    }


@app.post("/ingest")
def ingest():
    """Chunk every document in docs/ and upsert the chunks."""
    result = rag.ingest_documents(DOCS_PATH)
    return {
        "message": f"Indexed {result['chunks']} chunks from {result['documents']} document(s)",
        "documents": result["documents"],
        "chunks": result["chunks"],
        "files": result["files"],
        "stored": result["stored"],
    }

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str


@app.post("/ask")
def ask(req: AskRequest):
    """Answer from the corpus, with sources and confidence."""
    return rag.answer_question(req.question)


@app.get("/stats")
def stats():
    """Collection and configuration summary."""
    return {
        "document_count": rag.collection.count(),
        "collection": settings.COLLECTION_NAME,
        "model": settings.MODEL_NAME,
    }
