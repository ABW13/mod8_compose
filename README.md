# RAG API + Ollama + Streamlit — Docker Compose Stack

A three-service containerized RAG application: a FastAPI backend, a Streamlit
frontend, and Ollama for generation. All settings come from environment
variables.

## Structure

compose-demo/
├── docker-compose.yml
├── .env                 (not committed)
├── .env.example         (committed)
├── .gitignore
├── .dockerignore
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   └── docs/
│       └── sample.txt
└── frontend/
    ├── Dockerfile
    ├── app.py
    └── requirements.txt

## Run

    docker compose up --build
    docker compose exec ollama ollama pull llama3.2:1b

Then open [localhost](http://localhost:8501)

## Services

| Service  | Port  | Role                                  |
|----------|-------|---------------------------------------|
| backend  | 8000  | FastAPI RAG API, ChromaDB client      |
| frontend | 8501  | Streamlit chat UI                     |
| ollama   | 11434 | Model server (generation)             |

The frontend reaches the backend at [backend](http://backend:8000) and the backend reaches
Ollama at [ollama](http://ollama:11434) — both by Compose service name. Inside a
container, localhost means that container, so service names are the only
correct way to address a sibling.

## Endpoints

    GET  /               service name and configured model
    GET  /health         status, Ollama reachability, document count
    POST /ingest         index every .txt in docs/ (idempotent upsert)
    POST /ask            body {"question": "..."} -> answer + sources + confidence

## Configuration

config.py is a Settings class reading every value from the environment with
defaults. Values live in .env:

    OLLAMA_URL=[ollama](http://ollama:11434)
    MODEL_NAME=llama3.2:1b
    CHROMA_PATH=/app/chroma_data
    COLLECTION_NAME=documents
    MAX_RESULTS=5
    CONFIDENCE_THRESHOLD=1.0
    DEBUG=false
    DOCS_DIRECTORY=/app/docs
    OPENAI_API_KEY=

Because docker-compose.yml uses env_file: .env, changing a setting and running
`docker compose up -d` is enough — no image rebuild. Verified: MODEL_NAME was
changed to llama3.2:3b and the root endpoint reflected it after a restart with
no --build.

.env is gitignored, so a fresh clone has none and every setting falls back to
the config.py defaults. Note that the default OLLAMA_URL is
[localhost](http://localhost:11434) which inside a container means the container itself.
Copy .env.example to .env and set OLLAMA_URL=[ollama](http://ollama:11434)

MAX_RESULTS and CONFIDENCE_THRESHOLD are loaded and used by /ask; they cap the
number of retrieved chunks and set the confidence label.

## Persistence

Two named volumes survive `docker compose down`:

    chroma_data    the ChromaDB store
    ollama_data    pulled model weights

Observed readings:

| Step                | ollama    | documents |
|---------------------|-----------|-----------|
| after /ingest       | connected | 1         |
| after down + up     | connected | 1         |
| after down -v + up  | connected | 0         |

The count holding across down/up, then resetting after down -v, is the volume
evidence. The third reading rules out a container that never restarted.

Note: `down -v` also deletes the Ollama model. Re-pull after a volume reset.

## Version pins

    fastapi==0.109.0
    uvicorn==0.27.0
    chromadb==1.5.9
    requests==2.31.0

chromadb==1.5.9 is deliberate. main.py creates the collection with
configuration={"hnsw": {"space": "cosine"}}, and that parameter exists only
from chromadb 0.5.4 onward — it was added to the create-collection model in
that release. Chroma 1.0 made configuration the documented way to set index
options, keeping metadata={"hnsw:space": "cosine"} for backwards compatibility
only.

The common tutorial pin, chromadb==0.4.22, predates the parameter and raises
TypeError at import — before uvicorn binds port 8000, which makes it look like
a networking fault. 1.5.9 also requires pydantic >= 2.0, matching
fastapi==0.109.0.

Frontend:

    streamlit==1.67.0
    numpy==2.4.6
    pyarrow==26.0.1
    requests==2.31.0

Streamlit was originally pinned to 1.31.0, which failed: pip resolved a
pyarrow requiring NumPy 2.0+ while capping numpy<2, so pyarrow could not
import and Streamlit exited 1 before binding port 8501. The pins above are the
set pip resolved once the constraint was removed, captured with `pip freeze`.

Caveat: chromadb 1.5.9 carries four open CVE advisories with no published fix
versions. Every 1.x release currently carries the same set.

## Healthcheck

    test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('[localhost](http://localhost:8000/health)', timeout=5).status == 200 else 1)"]

The python:3.11-slim image has no curl, so a curl-based check fails with
"executable not found" on every attempt. Python's urllib needs nothing extra.
The check queries localhost from inside the backend container — its own server.

## Troubleshooting

Docker Desktop on WSL: if `docker` commands fail with "Failed to initialize:
protocol not available", the /var/run/docker.sock symlink may point at a socket
with no listener while the live Engine API sits elsewhere. Identify it by
probing each candidate:

    for s in /mnt/wsl/docker-desktop/shared-sockets/host-services/*.sock; do
      printf "%s -> " "$(basename "$s")"
      curl -s --max-time 2 --unix-socket "$s" [localhost](http://localhost/_ping) echo
    done

The one answering OK is the Engine API. Point a context at it:

    docker context create direct --docker "host=unix://<that socket path>"
    docker context use direct

This path lives under /mnt/wsl/, which Docker Desktop rebuilds on restart, so
the workaround may need reapplying after a reboot.
