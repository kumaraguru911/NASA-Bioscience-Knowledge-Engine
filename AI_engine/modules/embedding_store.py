# ai_engine/modules/embedding_store.py

"""
Advanced Embedding Store for NASA Bioscience Knowledge Engine.

Responsibilities:
- Generate embeddings from processed chunks (data_processor output)
- Build and manage FAISS index for semantic search
- Save/load FAISS index and metadata
- Provide semantic search with optional metadata filtering
- Support both persistent (on-disk) and session-based (in-memory) stores
"""

import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# ------------------ CONFIG ------------------
from AI_engine.config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    FAISS_INDEX_PATH, METADATA_PATH, LOG_LEVEL
)

# ------------------ LOGGER ------------------
logger = logging.getLogger("AI_engine.embedding_store")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# ------------------ EMBEDDING MODEL ------------------
embedding_model = None
openai_client = None

try:
    from sentence_transformers import SentenceTransformer
    import torch

    # Determine device: use HF_DEVICE from .env if set and valid, otherwise auto-detect.
    hf_device_setting = os.getenv("HF_DEVICE", "auto").lower()
    if hf_device_setting == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    logger.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL} on device: {device}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.warning(f"SentenceTransformer not available: {e}")
    try:
        import openai
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set; OpenAI embeddings cannot be used.")
        else:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"OpenAI embeddings will be used as fallback ({OPENAI_MODEL}).")
    except Exception as e2:
        logger.error(f"Failed to setup OpenAI embeddings: {e2}")

# ------------------ GET EMBEDDINGS ------------------
def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for texts using the primary model or OpenAI fallback.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)  # all-mpnet-base-v2 dim

    if embedding_model:
        return embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    elif openai_client:
        logger.info(f"Using OpenAI fallback to embed {len(texts)} texts...")
        response = openai_client.embeddings.create(model=OPENAI_MODEL, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)
    else:
        raise RuntimeError("No embedding method available. Install SentenceTransformers or provide OpenAI API key.")

# ------------------ BUILD STORE ------------------
def build_store(chunks: List[Dict], save_to_disk: bool = True) -> Tuple[Optional[faiss.Index], Optional[List[Dict]]]:
    """
    Build FAISS index from chunk embeddings. Optionally save index + metadata.
    """
    if not chunks:
        logger.warning("No chunks provided to build store.")
        return None, None

    try:
        texts = [c.get("chunk", "") for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = get_embeddings(texts)

        dim = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        ids = np.arange(len(chunks), dtype=np.int64)
        index.add_with_ids(embeddings.astype("float32"), ids)
        logger.info(f"FAISS index built with {index.ntotal} vectors.")

        if save_to_disk:
            logger.info(f"Saving index to {FAISS_INDEX_PATH} and metadata to {METADATA_PATH}.")
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(chunks, f)
            logger.info("Store saved successfully.")

        return index, chunks
    except Exception as e:
        logger.exception(f"Error building embedding store: {e}")
        return None, None

# ------------------ LOAD STORE ------------------
def load_store(index_path: str = FAISS_INDEX_PATH, metadata_path: str = METADATA_PATH) -> Tuple[Optional[faiss.Index], Optional[List[Dict]]]:
    """
    Load FAISS index and metadata from disk.
    """
    if not Path(index_path).exists() or not Path(metadata_path).exists():
        logger.warning("Index or metadata file not found. Build store first.")
        return None, None
    try:
        logger.info(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(str(index_path))
        logger.info(f"Loading metadata from {metadata_path}...")
        with open(Path(metadata_path), "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Embedding store loaded successfully with {index.ntotal} vectors.")
        return index, chunks
    except Exception as e:
        logger.exception(f"Failed to load embedding store: {e}")
        return None, None

# ------------------ SEARCH STORE ------------------
def search_store(query: str, index: faiss.Index, chunks: List[Dict], top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """
    Perform semantic search with optional metadata filtering.
    """
    if not query or index is None or not chunks:
        logger.warning("Invalid search inputs.")
        return []

    try:
        search_params = None
        if metadata_filter:
            valid_ids = []
            for i, chunk in enumerate(chunks):
                metadata_to_check = {**chunk, **chunk.get("paper_metadata", {})}
                if all(metadata_to_check.get(key) == value for key, value in metadata_filter.items()):
                    valid_ids.append(i)
            if not valid_ids:
                logger.warning("No documents match the metadata filter.")
                return []
            selector = faiss.IDSelectorArray(np.array(valid_ids, dtype=np.int64))
            search_params = faiss.SearchParametersIVF(sel=selector)

        query_vec = get_embeddings([query])
        distances, indices = index.search(query_vec.astype("float32"), top_k, params=search_params)
        valid_indices = [i for i in indices[0] if i != -1]
        results = [chunks[i] for i in valid_indices]

        logger.info(f"Query '{query}' returned {len(results)} results.")
        return results

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        return []

# ------------------ MAIN TEST ------------------
if __name__ == "__main__":
    from AI_engine.modules.nasa_api import get_plain_papers_for_pipeline
    from AI_engine.modules.data_processor import process_papers

    query = input("Enter NASA query: ").strip()
    raw_papers = get_plain_papers_for_pipeline(query, max_results=5)
    processed_chunks = process_papers(raw_papers)

    index, chunks = build_store(processed_chunks, save_to_disk=True)

    if index:
        search_query = input("Enter search query for retrieval: ").strip()
        top_results = search_store(search_query, index, chunks, top_k=3)
        for i, res in enumerate(top_results, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Paper: {res['title']}")
            print(f"Chunk: {res['chunk'][:300]}...")
