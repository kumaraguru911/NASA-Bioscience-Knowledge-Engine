# ai_engine/modules/rag_engine.py
"""
Optimized RAG Engine for NASA Bioscience Knowledge Engine
- FAISS retrieval with mpnet embeddings
- Contriever reranking
- GPT-4o-mini answer generation
- Citation-aware context
- Fallback extractive answer
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from AI_engine.config import TOP_K_RETRIEVAL, OPENAI_MODEL, USE_LLM, HUGGING_FACE_TOKEN
from AI_engine.modules.embedding_store import load_store, get_embeddings
from AI_engine.modules.nasa_api import get_plain_papers_for_pipeline
import torch
from transformers import pipeline
from AI_engine.utils.logger import get_logger

logger = get_logger('AI_engine.rag_engine')

# ----------------- Reranker setup -----------------
RERANKER_MODEL = 'BAAI/bge-reranker-base'
reranker_model = None
device = "cpu" # Default device
try:
    # Determine device: use HF_DEVICE from .env if set and valid, otherwise auto-detect.
    hf_device_setting = os.getenv("HF_DEVICE", "auto").lower()
    if hf_device_setting == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from sentence_transformers.cross_encoder import CrossEncoder
    reranker_model = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)
    logger.info(f"Reranker model '{RERANKER_MODEL}' loaded on device: {device}")
except Exception as e:
    logger.warning(f"Failed to load reranker model: {e}. Reranking will be skipped.")
    device = "cpu"

# ----------------- Local LLM Generator Setup -----------------
generator_pipeline = None
if USE_LLM:
    try:
        logger.info(f"Loading local LLM for generation: {OPENAI_MODEL} on device: {device}")
        generator_pipeline = pipeline(
            "text-generation",
            model=OPENAI_MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16}, # Use bfloat16 for better performance
            device=device,
            token=HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN else None,
        )
        logger.info("Local LLM generator loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load local LLM generator: {e}. Generation will be disabled.")

# ----------------- Utility functions -----------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not reranker_model:
        logger.debug("Reranker model not available, returning top_k chunks without reranking.")
        return chunks[:top_k]

    pairs = [(query, c['chunk']) for c in chunks]
    scores = reranker_model.predict(pairs, show_progress_bar=False)

    # attach score and sort
    for c, s in zip(chunks, scores):
        c['score'] = s
    
    sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
    return sorted_chunks[:top_k]

# ----------------- Build context for LLM -----------------
MAX_CONTEXT_CHARS = 3500
def build_context(query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    lines, source_labels = [], []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c.get('title', 'Unknown Title')} (ID:{c.get('paper_id', 'UnknownID')})"
        source_labels.append(label)
        snippet = c['chunk'][:1000]
        lines.append(f"Source [{i}]: {label}\nContent: {snippet}\n")
    context = '\n\n'.join(lines)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + '\n\n[context truncated]\n'
    return context, source_labels

# ----------------- LLM Answer Generation -----------------
def generate_with_local_llm(system_prompt: str, user_prompt: str) -> str:
    if not generator_pipeline:
        raise RuntimeError("Local LLM generator not initialized.")
    
    # Mistral instruction format
    prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    
    sequences = generator_pipeline(
        prompt,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=generator_pipeline.tokenizer.eos_token_id,
        max_new_tokens=1024,
    )
    return sequences[0]['generated_text'].split('[/INST]')[-1].strip()

# ----------------- Fallback extractive -----------------
def generate_fallback(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant documents found."
    lines = [f"Query: {query}\nSummary from retrieved NASA sources:\n"]
    for i, c in enumerate(chunks, 1):
        snippet = c['chunk'][:800].strip()
        lines.append(f"[{i}] {c.get('title', 'Unknown')} (ID:{c.get('paper_id', 'Unknown')})\n{snippet}\n")
    lines.append("\nNote: These are verbatim grounded snippets.")
    return '\n'.join(lines)

# ----------------- Run RAG Pipeline -----------------
def run_rag_pipeline(query: str, top_k: int = TOP_K_RETRIEVAL, index: Optional[Any] = None, chunks: Optional[List[Dict]] = None) -> Dict[str, Any]:
    logger.info(f"Running RAG pipeline for query: '{query}'")
    result = {"query": query, "answer": None, "retrieved_chunks": [], "debug": {}}

    # If no temporary store is passed, load the default persistent one
    if index is None or chunks is None:
        logger.info("Loading persistent knowledge base...")
        index, chunks = load_store()

    if not index or not chunks:
        result['answer'] = "Knowledge base unavailable."
        return result

    # Retrieve top-k using FAISS
    query_vec = get_embeddings([query])
    distances, indices = index.search(query_vec.astype('float32'), top_k)
    retrieved = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            retrieved.append({
                'chunk': chunks[idx]['chunk'],
                'title': chunks[idx].get('title', ''),
                'paper_id': chunks[idx].get('paper_id', ''),
                'metadata': chunks[idx].get('metadata', {}),
                'score': float(1 / (1 + distances[0][i]))
            })

    # Rerank the retrieved chunks
    top_chunks = rerank_chunks(query, retrieved, top_k)
    result['retrieved_chunks'] = top_chunks

    # Build context
    context, source_labels = build_context(query, top_chunks)
    system_prompt = (
        "You are a scientific research assistant. Answer using ONLY the provided Sources. "
        "Include inline citations like [1], [2]. If info missing, state so."
    )
    user_prompt = f"Question: {query}\nSources:\n{context}\nInstructions: Provide concise, citation-based answer."

    # Generate answer
    try:
        if USE_LLM and generator_pipeline:
            answer = generate_with_local_llm(system_prompt, user_prompt)
        else:
            answer = generate_fallback(query, top_chunks)
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}")
        answer = generate_fallback(query, top_chunks)

    result['answer'] = answer
    logger.info("RAG pipeline finished.")
    return result

# ----------------- CLI Test -----------------
if __name__ == '__main__':
    q = input("Enter your query: ").strip()
    res = run_rag_pipeline(q)
    print("\n--- ANSWER ---\n")
    print(res['answer'])
    print("\n--- SOURCES ---\n")
    for c in res['retrieved_chunks']:
        print(f"[{c['score']:.3f}] {c['title']} (id:{c['paper_id']})")
