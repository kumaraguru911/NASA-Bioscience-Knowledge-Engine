"""
Project Configuration - NASA Bioscience Knowledge Engine

Centralized configuration for:
- API keys
- Paths
- Model choices
- RAG and embedding parameters
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------ LOAD ENV ------------------
# Loads environment variables from .env in project root
AI_ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AI_ENGINE_DIR.parent
load_dotenv(dotenv_path=AI_ENGINE_DIR / ".env")

# ------------------ API KEYS ------------------
NASA_API_KEY = os.getenv("NASA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")

# ------------------ DATA PATHS ------------------
DATA_DIR = AI_ENGINE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------ EMBEDDING / RAG CONFIG ------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # SentenceTransformer model
USE_LLM = True                          # Whether to use a local LLM for generation
OPENAI_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" # Local model for RAG, Summarization, etc.
TOP_K_RETRIEVAL = 5                     # How many top chunks to retrieve
TEMPERATURE = 0.2                        # LLM answer temperature (low = more factual)

# ------------------ OTHER SETTINGS ------------------
MAX_PAPERS_FETCH = 20                   # Max number of NASA papers to fetch per query
CHUNK_SIZE = 500                         # Tokens/words per processed chunk
CHUNK_OVERLAP = 50                       # Overlap between chunks for context continuity
LOG_LEVEL = "INFO"                        # Logging level
