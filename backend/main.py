import os
from pathlib import Path

import chromadb
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration comes from the environment, set in docker-compose.yml.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL_NAME", "llama3.2:1b")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "/app/chroma_data")
DOCS_PATH = Path(os.environ.get("DOCS_PATH", "/app/docs"))

# PersistentClient writes to a directory; in Compose that directory is a
# named volume, which is what makes the data survive `docker-compose down`.
client = chromadb.PersistentClient(path=CHROMA_PATH)

# `configuration` is the current API (Chroma 1.0+). The pre-1.0 form was
# metadata={"hnsw:space": "cosine"}, which newer releases still accept for
# backwards compatibility but no longer document.
collection = client.get_or_create_collection(
    "documents", configuration={"hnsw": {"space": "cosine"}}
)


@app.get("/")
def root():
    return {"message": "RAG API running in Docker", "model": MODEL}


@app.get("/health")
def health():
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception as exc:
        print(f"Ollama health check failed for {OLLAMA_URL!r}: {exc}")
    return {
        "status": "healthy",
        "ollama": "connected" if ollama_ok else "unavailable",
        "ollama_url": OLLAMA_URL,
        "documents": collection.count(),
    }


@app.post("/ingest")
def ingest():
    """Seed the collection with one document per file in docs/.

    This exists so the persistence test is meaningful: `collection.count()`
    has to be non-zero before a restart for "the count was preserved" to
    demonstrate anything. Idempotent -- re-running replaces existing ids.
    """
    ingested = []
    if DOCS_PATH.is_dir():
        for path in sorted(DOCS_PATH.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            doc_id = f"doc:{path.name}"
            collection.upsert(ids=[doc_id], documents=[text])
            ingested.append({"id": doc_id, "chars": len(text)})

    return {"ingested": len(ingested), "files": ingested, "documents": collection.count()}
