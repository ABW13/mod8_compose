import os


class Settings:
    """Application settings loaded from environment variables."""

    # Ollama configuration
    OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "[localhost](http://localhost:11434)")
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama3.2:1b")

    # ChromaDB configuration
    CHROMA_PATH: str = os.environ.get("CHROMA_PATH", "./chroma_data")
    COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "documents")

    # RAG configuration
    MAX_RESULTS: int = int(os.environ.get("MAX_RESULTS", "3"))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "1.2"))

    # Application settings
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    DOCS_DIRECTORY: str = os.environ.get("DOCS_DIRECTORY", "./docs")

    # Optional cloud API key (for OpenAI fallback)
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")


settings = Settings()

if settings.DEBUG:
    print("=== Configuration ===")
    print(f"  Ollama URL: {settings.OLLAMA_URL}")
    print(f"  Model: {settings.MODEL_NAME}")
    print(f"  ChromaDB: {settings.CHROMA_PATH}")
    print(f"  Max Results: {settings.MAX_RESULTS}")
    print(f"  Threshold: {settings.CONFIDENCE_THRESHOLD}")
    print(f"  Debug: {settings.DEBUG}")
    print(f"  OpenAI key: {'set' if settings.OPENAI_API_KEY else 'not set'}")
