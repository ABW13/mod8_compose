# RAG API + Ollama — Multi-Service Docker Compose Stack

FastAPI backend and Ollama running as two connected containers on the same
Compose network.

## Structure

compose-demo/
├── docker-compose.yml
├── .dockerignore
├── README.md
└── backend/
    ├── Dockerfile
    ├── main.py
    ├── requirements.txt
    └── docs/
        └── sample.txt

## Run

    cd compose-demo
    docker compose up --build

Older standalone Compose uses the hyphenated `docker-compose up --build`. Both
forms work; the space form is Compose v2.

Wait until both services report ready, then in another terminal:

    curl [localhost](http://localhost:8000/)                 # root message
    curl [localhost](http://localhost:8000/health)           # "ollama": "connected"
    curl -X POST [localhost](http://localhost:8000/ingest)   # seed the collection
    curl [localhost](http://localhost:8000/health)           # "documents": 1

Browser docs: [localhost](http://localhost:8000/docs)

### Pull the model

The `ollama/ollama` image starts with no models downloaded. `MODEL_NAME` is only
a string the app holds — nothing fetches the weights. Pull it once, into the
`ollama_data` volume:

    docker compose exec ollama ollama pull llama3.2:1b

Expect about 1.3 GB. It survives restarts because the volume does.

`/health` reports `"ollama": "connected"` as soon as the Ollama server answers
its HTTP API, which happens before any model exists. A `connected` reading is not
proof the model is present.

## Persistence test

`/ingest` exists so this test means something. Without an ingest route,
`collection.count()` is 0 before and after a restart — zero equals zero, which
demonstrates nothing.

Observed readings:

| Step                | ollama    | documents |
|---------------------|-----------|-----------|
| after /ingest       | connected | 1         |
| after down + up     | connected | 1         |
| after down -v + up  | connected | 0         |

    curl -X POST [localhost](http://localhost:8000/ingest)     # documents: 1
    docker compose down                            # stop, KEEP volumes
    docker compose up -d
    curl [localhost](http://localhost:8000/health)              # documents: 1  <- survived

    docker compose down -v                         # delete volumes
    docker compose up -d
    curl [localhost](http://localhost:8000/health)              # documents: 0

The count holding across down/up, then resetting after down -v, is the volume
evidence. The third reading is what rules out a container that never actually
restarted — a preserved count alone could be a process that never stopped.

## Version pin rationale

    fastapi==0.109.0
    uvicorn==0.27.0
    chromadb==1.5.9
    requests==2.31.0

`chromadb==1.5.9` is deliberate. `main.py` creates the collection with
`configuration={"hnsw": {"space": "cosine"}}`, and that parameter only exists
from chromadb 0.5.4 onward — it was added to the create-collection model in that
release. Chroma 1.0 then made `configuration` the documented way to set index
options, with the older `metadata={"hnsw:space": "cosine"}` form kept for
backwards compatibility only.

The common tutorial pin, `chromadb==0.4.22`, predates the parameter entirely and
raises TypeError at import time — before uvicorn ever binds port 8000, which
makes it look like a networking fault when it is not.

1.5.9 also requires pydantic >= 2.0, which is what `fastapi==0.109.0` expects, so
the two agree rather than conflict. Chroma requires Python >= 3.9 and
`python:3.11-slim` sits inside that range.

Caveat: chromadb 1.5.9 carries four open CVE advisories (pip-audit will flag
them, with no fix versions published). Every 1.x release currently carries the
same set. Recorded here rather than left to be discovered.

## Design notes

Layer caching. `requirements.txt` is copied and installed before `COPY . .`, so
pip only re-runs when requirements change. The build log shows CACHED on steps
3-5 with only `COPY . .` re-running — the requirements-first ordering doing its
job.

Service-name resolution. The backend reaches Ollama at `[ollama](http://ollama:11434)`,
not `localhost`. On a Compose network, service names resolve as hostnames;
`localhost` inside the backend container would mean the backend container
itself. This is why no `host.docker.internal` and no
`--add-host=host.docker.internal:host-gateway` are needed here — that trick
exists to reach out of Docker, and running Ollama inside Compose removes the
need for it.

`depends_on` orders startup only. It does not wait for Ollama to be ready to
serve. That is why `/health` reports Ollama's state rather than assuming it: the
backend starts fine either way, and the endpoint tells you the truth at the
moment you ask. On the first run this produced an `unavailable` reading from a
startup race, which resolved on the next call.

Chroma's distance space is fixed at creation. `cosine` vs `l2` cannot be changed
afterwards without rebuilding the collection; setting it on an existing
collection is silently ignored.

`version:` is omitted. Compose v2 ignores it and prints an obsolescence warning.

The health check reports failures. The `except` clause prints the URL that
failed and the exception, so a malformed `OLLAMA_URL` surfaces in
`docker compose logs backend` instead of going quiet as a bare `unavailable`.
