# ai_engine/modules/data_processor.py

"""
Professional preprocessing module for NASA Bioscience Knowledge Engine.

Responsibilities:
- Accepts raw papers fetched from NASA APIs (NTRS or ADS)
- Cleans and normalizes text dynamically
- Uses NLTK for tokenization and lemmatization
- Splits text into meaningful chunks (~400-500 tokens)
- Extracts dynamic entities, keywords, and headings using spaCy/SciSpacy
- Returns structured JSON-ready chunks for embedding/RAG pipelines
"""

import os
import re
import logging
import unicodedata
from typing import List, Dict, Any

# ------------------ NLTK Setup ------------------
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
for pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg=="punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ------------------ SpaCy Setup ------------------
import spacy
try:
    nlp = spacy.load("en_core_sci_sm")  # SciSpacy for biomedical text
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")  # fallback
    except Exception:
        nlp = None

# ------------------ Logger ------------------
logger = logging.getLogger("AI_engine.data_processor")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ------------------ Helper: Text Cleaning ------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    # text = re.sub(r"\d+", "", text) # Keep numbers as they can be scientifically relevant
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.lower()

# ------------------ Helper: Lemmatization & Stopword removal ------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text: str) -> str:
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 2]
    return " ".join(filtered)

# ------------------ Helper: Chunking ------------------
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_len = 0

    for sent in sentences:
        token_count = len(sent.split())
        if current_len + token_count > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = sent.split()
            current_len = token_count
        else:
            current_chunk.extend(sent.split())
            current_len += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ------------------ Helper: Dynamic metadata ------------------
def extract_dynamic_metadata(text: str, top_k_keywords: int = 10) -> Dict[str, Any]:
    if not text or nlp is None:
        return {"text": text, "entities": [], "keywords": [], "headings": []}
    try:
        doc = nlp(text)
    except Exception as e:
        logger.warning(f"SpaCy processing failed: {e}")
        return {"text": text, "entities": [], "keywords": [], "headings": []}

    entities = [ent.text for ent in doc.ents if ent.text.strip()]
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]

    from collections import Counter
    if entities:
        keywords = [k for k, _ in Counter(entities).most_common(top_k_keywords)]
    else:
        keywords = list(dict.fromkeys(noun_chunks))[:top_k_keywords]

    headings = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) < 200 and (":" in sent.text or sent.text.isupper())]

    return {"text": text, "entities": entities, "keywords": keywords, "headings": headings}

# ------------------ Main processing: papers -> chunks ------------------
def process_papers(raw_papers: List[Dict[str, Any]], top_k_keywords: int = 10) -> List[Dict[str, Any]]:
    processed = []
    for paper in raw_papers:
        paper_id = paper.get("id", "UnknownID")
        title = paper.get("title", "Unknown Title")
        pdf_link = paper.get("pdf_link")
        paper_metadata = paper.get("metadata", {})

        text_to_process = paper.get("summary") or ""
        text_to_process = clean_text(text_to_process)
        if not text_to_process:
            logger.debug(f"Skipping paper '{title}' (ID: {paper_id}) due to empty summary.")
            continue

        text_to_process = lemmatize_text(text_to_process)
        chunks = chunk_text(text_to_process)

        for chunk_text_item in chunks:
            chunk_meta = extract_dynamic_metadata(chunk_text_item, top_k_keywords=top_k_keywords)
            chunk_meta.update({
                "paper_id": paper_id,
                "title": title,
                "pdf_link": pdf_link,
                "paper_metadata": paper_metadata
            })
            chunk_meta["chunk"] = chunk_meta.pop("text")  # rename
            processed.append(chunk_meta)

    logger.info(f"Processed {len(processed)} chunks from {len(raw_papers)} papers.")
    return processed

# ------------------ __main__ test ------------------
if __name__ == "__main__":
    from AI_engine.modules.nasa_api import get_plain_papers_for_pipeline
    query = input("Enter NASA search query for processing: ").strip() 
    raw_papers = get_plain_papers_for_pipeline(query, max_results=5, source="NTRS")
    processed_chunks = process_papers(raw_papers)
    for i, c in enumerate(processed_chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Paper ID: {c['paper_id']}")
        print(f"Title   : {c['title']}")
        print(f"Chunk   : {c['chunk'][:300]}...")
        print(f"Keywords: {c['keywords']}")
        print(f"Entities: {c['entities']}")
        print(f"Headings: {c['headings']}")
