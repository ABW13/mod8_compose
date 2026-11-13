import os
import pathlib

os.environ.setdefault("CHROMA_PATH", "./test_chroma")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
# Absolute, so the test works from any working directory.
os.environ.setdefault(
    "DOCS_DIRECTORY",
    str(pathlib.Path(__file__).resolve().parent.parent / "docs"),
)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "model" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "documents" in data


def test_ingest():
    response = client.post("/ingest")
    assert response.status_code == 200
    data = response.json()
    assert data["ingested"] >= 1
    assert data["documents"] >= 1


def test_ask_returns_sources():
    client.post("/ingest")
    response = client.post("/ask", json={"question": "How does authentication work?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert data["confidence"] in {"low", "medium", "high"}


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
