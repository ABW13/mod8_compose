FROM python:3.11-slim

WORKDIR /app

# build-essential is needed for several ChromaDB dependencies that compile
# native extensions during install (tokenizers, grpcio, onnxruntime helpers).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so this layer is cached and pip only re-runs
# when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the docs/ folder used by /ingest
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
