# Containerized RAG Chatbot

A retrieval-augmented generation chatbot that runs entirely with
`docker compose up`. Documents are chunked and stored in ChromaDB, relevant
chunks are retrieved for each question, Ollama generates an answer grounded in
those chunks, and Streamlit presents the result with citations.

## Architecture

Three services on one Compose network, addressed by service name. The frontend
never touches ChromaDB directly; it reads the chunk count from the backend's
health endpoint.

| Connection | Value | Why |
|---|---|---|
| frontend to backend | Compose service name `backend:8000` | set in `API_URL` |
| backend to ollama | Compose service name `ollama:11434` | set in `OLLAMA_URL` |
| backend to ChromaDB | `PersistentClient` on `CHROMA_PATH` | mounted volume |
| backend to corpus | `DOCS_DIRECTORY` | read at ingest time |

## Request flow

1. `POST /ingest` reads every `.txt` in `docs/`, splits each into overlapping
   chunks, embeds them, and stores them with `source` and `chunk` metadata.
2. `POST /ask` embeds the question, retrieves the closest chunks, and checks the
   best distance against `CONFIDENCE_THRESHOLD`.
3. If the closest chunk is within the threshold, the chunks are assembled into a
   grounded prompt and sent to Ollama. Otherwise the question is refused.
4. The response carries the answer, the sources with their distances, and a
   confidence label.

## Prerequisites

Docker Desktop with the WSL 2 backend, and roughly 2 GB of free disk for the
model.

## Setup

    git clone git@github.com:ABW13/mod8_compose.git
    cd compose-demo
    cp .env.example .env
    docker compose up --build

Then pull the model. The Ollama image ships with no models, so this step is
required. Without it the server answers requests about its own status and
returns a not-found error when asked to generate.

    docker compose exec ollama ollama pull llama3.2:1b
    docker compose exec ollama ollama list

Then index the corpus, either from the sidebar button in the UI or directly:

    curl -X POST localhost:8000/ingest

## Usage

Open the Streamlit interface on port 8501. The sidebar shows API status, Ollama
state and the chunk count, with a Re-index button. Ask a question in the chat
box; the answer appears with a sources expander showing which chunk supported it
and how close the match was.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service name and configured model |
| `/health` | GET | Status, Ollama reachability, chunk count |
| `/stats` | GET | Collection and model summary |
| `/ingest` | POST | Chunk and index the corpus |
| `/ask` | POST | Answer a question, with sources and confidence |

Interactive API documentation is served by the backend on port 8000 at `/docs`.

## Configuration

Every setting comes from the environment; nothing is hardcoded. `config.py`
defines the `Settings` class, `.env` supplies the values, and
`docker-compose.yml` loads it with `env_file`.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | localhost port 11434 | Model server address |
| `MODEL_NAME` | llama3.2:1b | Model to generate with |
| `CHROMA_PATH` | ./chroma_data | Vector store directory |
| `COLLECTION_NAME` | documents | ChromaDB collection |
| `MAX_RESULTS` | 3 | Chunks retrieved per question |
| `CONFIDENCE_THRESHOLD` | 0.75 | Retrieval distance cutoff |
| `DEBUG` | false | Print configuration at startup |
| `DOCS_DIRECTORY` | ./docs | Corpus location |
| `OPENAI_API_KEY` | empty | Reserved for a cloud fallback |

Because the environment is applied when containers are created rather than when
images are built, changing a value and recreating the containers is enough. No
rebuild is needed.

`.env` is gitignored, so a fresh clone has none and every setting falls back to
the `config.py` defaults. The default `OLLAMA_URL` points at localhost, which
inside a container means the container itself. Copy `.env.example` to `.env` and
set the Compose service names.

## Chunking

Documents are split on paragraph boundaries, targeting 220 words per chunk with
50 words of overlap. Splitting on structure rather than a fixed character window
keeps each chunk semantically coherent, and the overlap means a passage spanning
a boundary appears intact in at least one chunk.

The corpus of 5 documents and 5,508 words produces 37 chunks: 7 each from the
ChromaDB, Docker Compose, FastAPI and Ollama guides, and 9 from the RAG concepts
guide.

Each chunk carries `source` and `chunk` metadata, which is what makes a citation
possible.

## Guardrail

ChromaDB returns cosine distance: smaller means closer. After retrieval the best
distance is compared against `CONFIDENCE_THRESHOLD`. If it exceeds the threshold
the question is refused, instead of being answered from unrelated text.

The threshold was set from measurement rather than guessed:

| Question | Answerable | Best distance | Result |
|---|---|---|---|
| What distance metric does ChromaDB use for text? | yes | 0.470 | answered |
| How should I choose a chunk size? | yes | 0.599 | answered |
| How far apart should my Docker services be started? | yes | 0.603 | answered |
| Who won the 2018 World Cup? | no | 0.891 | refused |
| What is the capital of France? | no | 0.897 | refused |
| How do I bake sourdough bread? | no | 0.971 | refused |

Answerable questions land at or below 0.603; unanswerable ones at or above
0.891, so the threshold sits at 0.75, roughly the middle of that gap.

Two caveats. The separation is a property of this corpus and would need
re-measuring on a different collection. And the same value drives the high and
medium labels, which bucket at threshold times 0.6, so changing it shifts both
the refusals and the labels.

## Tests

    cd backend
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest 'httpx<0.28'
    .venv/bin/python -m pytest tests/ -v

Five tests cover `/`, `/health`, `/ingest`, `/ask` and `/stats`. The ingest test
asserts that chunks outnumber documents, so it fails if chunking stops working.

CI runs on every push to main via `.github/workflows/ci.yml`: a test job, a job
that builds both images, and a ruff lint job. The image verification step uses
`docker image inspect`, which fails when an image is missing; `docker images`
does not, since it exits zero either way.

## Project layout

    compose-demo/
    |-- docker-compose.yml
    |-- .env                      (not committed)
    |-- .env.example
    |-- .github/workflows/ci.yml
    |-- pyproject.toml            ruff configuration
    |-- README.md
    |-- backend/
    |   |-- Dockerfile
    |   |-- main.py               FastAPI routes
    |   |-- rag.py                chunking, retrieval, generation, guardrail
    |   |-- config.py             environment-based settings
    |   |-- requirements.txt
    |   |-- docs/                 corpus, 5 documents
    |   `-- tests/test_api.py
    `-- frontend/
        |-- Dockerfile
        |-- app.py                Streamlit chat interface
        `-- requirements.txt

## Version pins

Backend: fastapi 0.109.0, uvicorn 0.27.0, chromadb 1.5.9, requests 2.31.0.
Frontend: streamlit 1.67.0, numpy 2.4.6, pyarrow 26.0.1, requests 2.31.0.

`chromadb==1.5.9` is deliberate: the collection is created with the
configuration dictionary carrying an hnsw space of cosine, and that parameter
exists only from chromadb 0.5.4 onward. The commonly cited 0.4.22 predates it and
raises at import, before the server binds its port, which makes it look like a
networking fault.

The frontend pins are the set pip resolved once the original streamlit 1.31.0
constraint was removed. That version pulled a pyarrow requiring NumPy 2.0 or
newer while capping numpy below 2, so Streamlit could not import.

Test dependencies are pinned too: httpx below 0.28, because newer versions
removed the client API that Starlette's TestClient wraps.

## Persistence

Two named volumes survive a plain `docker compose down`:

| Volume | Contents |
|---|---|
| `chroma_data` | the vector store |
| `ollama_data` | pulled model weights |

`docker compose down -v` deletes both. After it, re-pull the model and re-index
the corpus.

## What I would improve

- Chunking is paragraph-based and fixed; sentence-aware or heading-aware
  splitting would likely improve retrieval on longer documents.
- One embedding model serves everything; a code-specific model would suit a
  technical corpus better.
- The threshold is tuned to this corpus; measuring it automatically or adding a
  reranking step would make the guardrail portable.
- No conversation memory, so follow-up questions cannot refer to earlier turns.
- Retrieval is single-shot; query expansion or a second pass would help when the
  first attempt misses.
