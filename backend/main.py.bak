from pathlib import Path

import chromadb
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

DOCS_PATH = Path(settings.DOCS_DIRECTORY)

# PersistentClient writes to a directory; in Compose that directory is a
# named volume, which is what makes the data survive `docker-compose down`.
client = chromadb.PersistentClient(path=settings.CHROMA_PATH)

# `configuration` is the current API (Chroma 1.0+). The pre-1.0 form was
# metadata={"hnsw:space": "cosine"}, which newer releases still accept for
# backwards compatibility but no longer document.
collection = client.get_or_create_collection(
    settings.COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
)


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
            collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[{"source": path.name}],
            )
            ingested.append({"id": doc_id, "chars": len(text)})

    return {
        "message": f"Indexed {len(ingested)} document(s)",
        "ingested": len(ingested),
        "files": ingested,
        "documents": collection.count(),
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    """Retrieve relevant chunks, then ask Ollama to answer from them."""
    results = collection.query(
        query_texts=[req.question], n_results=settings.MAX_RESULTS
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    sources = []
    for i, text in enumerate(docs):
        meta = metas[i] or {} if i < len(metas) else {}
        sources.append(
            {
                "source": meta.get("source", f"document-{i}"),
                "distance": float(dists[i]) if i < len(dists) else 0.0,
                "text": text,
            }
        )

    context = "\n\n".join(docs) if docs else ""
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}"
    )

    try:
        r = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={"model": settings.MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        answer = r.json().get("response", "").strip() or "No answer produced."
    except Exception as exc:
        print(f"Ollama generate failed for {settings.OLLAMA_URL!r}: {exc}")
        answer = f"Could not reach the model: {exc}"

    if not sources:
        confidence = "low"
    elif min(s["distance"] for s in sources) < settings.CONFIDENCE_THRESHOLD:
        confidence = "high"
    else:
        confidence = "medium"

    return {"answer": answer, "sources": sources, "confidence": confidence}


@app.get("/stats")
def stats():
    return {
        "document_count": collection.count(),
        "collection": settings.COLLECTION_NAME,
        "model": settings.MODEL_NAME,
    }
