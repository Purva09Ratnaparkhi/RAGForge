"""
RAGForge — Central Configuration
=================================

Every tunable value in the project lives here as a module-level constant.
Other modules import from ``config.settings`` only — no magic numbers
anywhere else in the codebase.

Secrets (API keys) are loaded from a ``.env`` file via ``python-dotenv``
so they never appear in version control.

Sections
--------
* Directories — where raw PDFs, processed data, and FAISS indexes live
* LLM — provider selection, model names, generation parameters
* Embeddings — local HuggingFace BGE model settings
* Chunking — chunk size and overlap for the smart chunker
* Retrieval — top-k, weights, RRF constant
* Reranker — cross-encoder model name
* Extraction — quality thresholds for the PDF extraction pipeline
* Serving — FastAPI host/port, CORS origins
* RAGAS — evaluation sample size
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (must be at project root)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
RAW_PDF_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FAISS_STORE_DIR: Path = DATA_DIR / "faiss_store"
LOG_DIR: Path = ROOT_DIR / "logs"

# Ensure directories exist at import time so downstream code never has to
# worry about missing folders.
for _d in (RAW_PDF_DIR, PROCESSED_DIR, FAISS_STORE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = "groq"  # "groq" | "openai"
GROQ_MODEL: str = "llama-3.3-70b-versatile"
OPENAI_MODEL: str = "gpt-4o"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024

# API keys — loaded from environment / .env
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE: str = "cpu"
EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 64
SEMANTIC_THRESHOLD: float = 0.85  # merge threshold for semantic chunking

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = 20
RERANK_TOP_K: int = 5
BM25_WEIGHT: float = 0.4
VECTOR_WEIGHT: float = 0.6
RRF_K: int = 60

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Extraction quality thresholds
# ---------------------------------------------------------------------------
MIN_CHARS_PER_PAGE: int = 100
GARBLE_RATIO_THRESHOLD: float = 0.15
HEADER_FOOTER_REPEAT_RATIO: float = 0.6
OCR_DPI: int = 300
OCR_DPI_RETRY: int = 400
UNSTRUCTURED_STRATEGY: str = "hi_res"

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
CORS_ORIGINS: list[str] = ["http://localhost:8501"]

# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------
RAGAS_SAMPLE_SIZE: int = 50
