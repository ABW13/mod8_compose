"""Retrieval-augmented generation pipeline.

Splits documents into overlapping chunks, stores them in ChromaDB with
provenance metadata, retrieves relevant chunks for a question, and asks
Ollama to answer from the retrieved context only.
"""

from pathlib import Path

import chromadb
import requests

from config import settings


# --- Chunking ---------------------------------------------------------------

# Chunks are built from whole paragraphs so a passage is never cut mid-sentence.
# A chunk targets CHUNK_WORDS but is allowed to run on until the next paragraph
# boundary, so the real upper bound is CHUNK_WORDS plus one paragraph.
CHUNK_WORDS = 220
OVERLAP_WORDS = 50


def iter_paragraphs(text: str):
    """Yield non-empty paragraphs from a document.

    A blank line separates paragraphs. This is the structural boundary in the
    corpus and splitting on it keeps each chunk semantically coherent, which a
    fixed character window would not.
    """
    for block in text.split("\n\n"):
        cleaned = " ".join(block.split())
        if cleaned:
            yield cleaned


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS,
               overlap_words: int = OVERLAP_WORDS):
    """Split text into overlapping chunks, respecting paragraph boundaries.

    Returns a list of chunk strings. Consecutive chunks share their last
    ``overlap_words`` words, so a passage spanning a boundary appears intact in
    at least one chunk rather than being split across two.
    """
    paragraphs = list(iter_paragraphs(text))
    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # Start a new chunk if adding this paragraph would overshoot the target
        # and we already have content.
        if current and current_words + para_words > chunk_words:
            chunks.append(" ".join(current))

            # Carry the tail of the finished chunk over as overlap.
            if overlap_words > 0:
                tail = " ".join(current).split()[-overlap_words:]
                current = [" ".join(tail)] if tail else []
                current_words = len(tail)
            else:
                current = []
                current_words = 0

        current.append(para)
        current_words += para_words

    if current:
        chunks.append(" ".join(current))

    return chunks


# --- Store ------------------------------------------------------------------

client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
collection = client.get_or_create_collection(
    settings.COLLECTION_NAME,
    configuration={"hnsw": {"space": "cosine"}},
)


def ingest_documents(docs_dir: Path, chunk_words: int = CHUNK_WORDS):
    """Chunk every .txt under docs_dir and upsert the chunks into ChromaDB.

    Each chunk is stored with metadata recording its source file and its index
    within that file, which is what makes a citation possible later. Ids are
    derived from both, so re-running is idempotent.
    """
    indexed = []
    seen_ids = set()

    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, chunk_words=chunk_words)

        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{path.name}:{i}"
            seen_ids.add(chunk_id)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {"source": path.name, "chunk": i, "total_chunks": len(chunks)}
            )

        if ids:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            indexed.append({"source": path.name, "chunks": len(chunks)})

    return {
        "documents": len(indexed),
        "chunks": sum(item["chunks"] for item in indexed),
        "files": indexed,
        "stored": collection.count(),
    }


# --- Retrieval and generation ----------------------------------------------

NO_CONTEXT_ANSWER = (
    "I could not find anything in the indexed documents that answers that "
    "question. Try rephrasing it, or ask about a topic the corpus covers."
)

PROMPT_TEMPLATE = """You are a documentation assistant. Answer the question \
using only the context provided below.

Rules:
- Base your answer solely on the context.
- If the context does not contain the answer, say so plainly rather than \
guessing.
- Be concise and specific.

Context:
{context}

Question: {question}

Answer:"""


def _confidence_label(best_distance: float | None, threshold: float) -> str:
    """Map the closest distance to a label.

    Chroma returns cosine *distance*: smaller is closer. A small value means a
    strong match, so the comparison is against the top of the range, not the
    bottom.
    """
    if best_distance is None:
        return "low"
    if best_distance < threshold * 0.6:
        return "high"
    if best_distance < threshold:
        return "medium"
    return "low"


def retrieve(question: str, n_results: int | None = None):
    """Return the closest chunks for a question, as a list of dicts."""
    n = n_results or settings.MAX_RESULTS
    total = collection.count()
    if total == 0:
        return []

    results = collection.query(query_texts=[question], n_results=min(n, total))

    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    sources = []
    for i, text in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        sources.append(
            {
                "source": meta.get("source", f"chunk-{i}"),
                "chunk": meta.get("chunk", i),
                "distance": float(distances[i]) if i < len(distances) else None,
                "text": text,
            }
        )
    return sources


def answer_question(question: str):
    """Retrieve context, ask Ollama, and return a structured response.

    Guardrail: if nothing was retrieved, or the closest match is further than
    CONFIDENCE_THRESHOLD, the question is refused instead of generating an
    answer from unrelated text. Distance is smaller-is-closer, so the check is
    a maximum, not a minimum.
    """
    sources = retrieve(question)
    distances = [s["distance"] for s in sources if s["distance"] is not None]
    best = min(distances) if distances else None

    threshold = settings.CONFIDENCE_THRESHOLD

    if not sources or best is None or best > threshold:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "sources": [],
            "confidence": "low",
            "guardrail": "rejected_below_threshold",
            "best_distance": best,
        }

    context = "\n\n---\n\n".join(s["text"] for s in sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except Exception as exc:
        print(f"Ollama generate failed for {settings.OLLAMA_URL!r}: {exc}")
        answer = f"Could not reach the model: {exc}"

    if not answer:
        answer = "The model returned an empty response."

    return {
        "answer": answer,
        "sources": sources,
        "confidence": _confidence_label(best, threshold),
        "guardrail": "answered",
        "best_distance": best,
    }
