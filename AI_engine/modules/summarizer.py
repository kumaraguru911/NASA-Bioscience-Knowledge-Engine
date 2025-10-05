# ai_engine/modules/summarizer.py
"""
State-of-the-art Summarization Module for NASA Bioscience Knowledge Engine.
Uses FLAN-T5 / LED for abstractive summarization + extractive fallback.
"""

import logging
import sys
import os
from typing import List, Dict, Any, Optional
from AI_engine.utils.logger import get_logger
import torch

from AI_engine.config import USE_LLM, OPENAI_API_KEY, OPENAI_MODEL, LOG_LEVEL, EMBEDDING_MODEL, DATA_DIR
from AI_engine.modules.data_processor import clean_text, extract_dynamic_metadata

logger = get_logger("AI_engine.summarizer", LOG_LEVEL)

# ------------------ MODEL SETUP ------------------
embedding_model = None
_SUMMARY_PIPELINE = None

# Setup SentenceTransformer for extractive fallback
try:
    from sentence_transformers import SentenceTransformer
    # Use the same device logic as other modules for consistency
    hf_device_setting_extractive = os.getenv("HF_DEVICE", "auto").lower()
    if hf_device_setting_extractive == "cuda" and torch.cuda.is_available():
        device_extractive = torch.device("cuda")
    else:
        device_extractive = torch.device("cpu")
        
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device_extractive)
    logger.info(f"SentenceTransformer '{EMBEDDING_MODEL}' loaded for extractive summarization on device: {device_extractive}.")
except Exception as e:
    logger.warning(f"Could not load SentenceTransformer model: {e}")

# Setup HuggingFace summarization pipeline (FLAN-T5 or LED)
# This pipeline is for the primary, non-LLM summarizer.
if True: # Always try to load the default summarizer
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

        SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "google/flan-t5-large")

        # Determine device: use HF_DEVICE from .env if set and valid, otherwise auto-detect.
        hf_device_setting = os.getenv("HF_DEVICE", "auto").lower()
        if hf_device_setting == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        _SUMMARY_PIPELINE = pipeline("summarization", model=SUMMARIZER_MODEL, device=device)
        set_seed(42)
        logger.info(f"HuggingFace summarizer pipeline loaded: {SUMMARIZER_MODEL} on device: {device}")
    except Exception as e:
        logger.warning(f"Could not load HF summarizer pipeline: {e}")

# ------------------ SUMMARIZATION ------------------
def summarize_extractive(text: str, num_sentences: int = 3) -> str:
    """Extractive summarization using sentence embeddings."""
    if not embedding_model:
        return ". ".join(text.split(". ")[:num_sentences]) + "."
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    sentences = [s.strip() for s in text.split('.') if len(s.strip())>10]
    if not sentences:
        return ""
    embeddings = embedding_model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    scores = sim_matrix.sum(axis=1) - 1
    top_idx = sorted(np.argsort(scores)[-num_sentences:])
    return " ".join([sentences[i] for i in top_idx]) + "."
def summarize_hf(text: str, max_chunk_tokens: int = 450, max_length: int = 150, min_length: int = 40) -> Optional[str]:
    """Abstractive summarization with HuggingFace LLM."""
    if not _SUMMARY_PIPELINE:
        return None
    cleaned_text = clean_text(text)
    # Chunk text to fit model context window (e.g., 512 for T5-base)
    words = cleaned_text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_tokens):
        chunks.append(" ".join(words[i:i+max_chunk_tokens]))
    
    if not chunks:
        return ""

    summaries = []
    for chunk in chunks:
        try:
            # Add a prompt for better focus
            prompt = f"Summarize the following research text: {chunk}"
            # Use max_new_tokens for more flexible output length control and remove max_length to avoid conflicts.
            out = _SUMMARY_PIPELINE(
                prompt, 
                max_new_tokens=max_length, # Re-purpose max_length for max_new_tokens
                min_length=min_length, 
                do_sample=False, 
                truncation=True
            )
            summaries.append(out[0]['summary_text'])
        except Exception as e:
            logger.warning(f"HF summarization failed on chunk: {e}")

    combined_summary = " ".join(summaries)
    
    # If we have multiple summaries, create a final "meta-summary" of them.
    if len(summaries) > 1:
        logger.info("Recursively summarizing the combined summary for conciseness.")
        final_summary_prompt = f"Create a final, coherent summary from the following points: {combined_summary}"
        # Ensure the final summary has a reasonable length.
        # Ensure max_length is always greater than min_length, even if the base max_length is small.
        final_max_length = max(max_length, min_length + 20)
        final_min_length = min(min_length, final_max_length - 10) # Ensure min_length is valid
        
        final_out = _SUMMARY_PIPELINE(
            final_summary_prompt, max_new_tokens=final_max_length, min_length=final_min_length, do_sample=False, truncation=True
        )
        return final_out[0]['summary_text']
    else:
        return combined_summary

def summarize_llm(text: str) -> Optional[str]:
    """OpenAI LLM summarization."""
    # This now uses the local LLM generator from rag_engine
    try:
        from .rag_engine import generate_with_local_llm, generator_pipeline
        if not generator_pipeline:
            logger.warning("Local LLM generator not available for summarization.")
            return None
    except ImportError:
        logger.warning("Could not import local LLM generator for summarization.")
        return None

    system_prompt = "You are a scientific editor. Summarize the research text concisely."
    user_prompt = f"Text:\n{text}"
    try:
        return generate_with_local_llm(system_prompt, user_prompt)
    except Exception as e:
        logger.warning(f"Local LLM summarization failed: {e}")
        return None

def summarize(text: str, method: str = "hf", num_sentences: int = 3) -> Dict[str, Any]:
    """Main summarization function with fallback strategies."""
    cleaned_text = clean_text(text)
    summary_text = ""
    used_method = ""

    # Preferred HF summarization
    if method == "hf" and _SUMMARY_PIPELINE:
        summary_text = summarize_hf(cleaned_text)
        used_method = "hf"

    # Fallback OpenAI LLM
    if not summary_text and USE_LLM:
        summary_text = summarize_llm(cleaned_text)
        used_method = "llm"

    # Fallback extractive
    if not summary_text:
        summary_text = summarize_extractive(cleaned_text, num_sentences=num_sentences)
        used_method = "extractive"

    keywords = extract_dynamic_metadata(summary_text, top_k_keywords=5).get("keywords", [])

    return {"summary": summary_text, "method": used_method, "keywords": keywords}
