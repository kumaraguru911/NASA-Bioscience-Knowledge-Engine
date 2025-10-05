"""
ai_engine/modules/nasa_api.py

Handles all communication with NASA's research APIs (NTRS, ADS).

- Fetches papers with advanced filtering (mission, organism, etc.).
- Caches responses locally for speed and offline use.
- Provides paginated fetching for large-scale data ingestion.
- Normalizes API responses for downstream processing.
- Uses a resilient HTTP session with automatic retries.
"""

from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import json
import hashlib
import time
import os
import sys

# Adjust path for standalone execution
if __name__ == '__main__' and __package__ is None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import from centralized config
from AI_engine.config import (
    NASA_API_KEY, OPENAI_API_KEY,
    DATA_DIR, RAW_DATA_DIR,
    MAX_PAPERS_FETCH, LOG_LEVEL
)

# Logger
logger = logging.getLogger("ai_engine.nasa_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# --- Constants ---
USER_AGENT = "NASA-Bioscience-Knowledge-Engine/1.0"
DEFAULT_TIMEOUT = 30
CACHE_DIR = RAW_DATA_DIR / "api_cache"
CACHE_DIR.mkdir(exist_ok=True)

# HTTP session with retry
def create_session(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session

SESSION = create_session()

# -------------------------
# Caching utilities
# -------------------------
def get_cache_key(prefix: str, params: Dict) -> str:
    """Generate a consistent hash key for caching."""
    # Use a stable representation of the dict for hashing
    encoded_params = json.dumps(params, sort_keys=True).encode('utf-8')
    return f"{prefix}_{hashlib.md5(encoded_params).hexdigest()}.json"

def read_from_cache(key: str) -> Optional[List[Dict]]:
    """Read API response from a local cache file."""
    cache_file = CACHE_DIR / key
    if cache_file.exists():
        logger.debug(f"Cache hit for key: {key}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    logger.debug(f"Cache miss for key: {key}")
    return None

def write_to_cache(key: str, data: List[Dict]):
    """Write API response to a local cache file."""
    cache_file = CACHE_DIR / key
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# -------------------------
# NTRS fetcher
# -------------------------
def fetch_from_ntrs(query: str, max_results: int, start: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch papers from NTRS API and return normalized paper dicts.
    """
    url = "https://ntrs.nasa.gov/api/citations/search"
    params = {"q": query, "start": start, "rows": max_results}
    papers: List[Dict[str, Any]] = []

    try:
        logger.info(f"Querying NTRS: q='{query}', rows={max_results}, start={start}")
        resp = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", []) if isinstance(data, dict) else []

        for idx, doc in enumerate(results[:max_results]):
            paper_id = doc.get("id") or doc.get("documentId") or f"NTRS_{start + idx}"
            title = doc.get("title") or doc.get("documentTitle") or "Unknown Title"
            summary = doc.get("abstract") or doc.get("description") or ""
            
            # Extract PDF link if available
            pdf_link = next((link.get("url") for link in doc.get("links", []) if isinstance(link, dict) and "pdf" in str(link.get("format", "")).lower()), None)

            # Keep all other fields as metadata
            metadata = {k: v for k, v in doc.items() if k not in {"id", "title", "abstract", "description", "links"}}

            paper = {
                "id": str(paper_id),
                "source": "NTRS",
                "title": title,
                "summary": summary,
                "pdf_link": pdf_link,
                "metadata": metadata
            }
            papers.append(paper)

    except requests.exceptions.RequestException as e:
        logger.error(f"NTRS request error: {e}")
    except ValueError as e:
        logger.error(f"NTRS JSON decode error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in fetch_from_ntrs: {e}")

    return papers

# -------------------------
# ADS fetcher
# -------------------------
def fetch_from_ads(query: str, max_results: int, start: int = 0) -> List[Dict[str, Any]]:
    """
    Fetch papers from ADS API. ADS key must be provided (either via env or ads_api_key).
    """
    if not NASA_API_KEY:
        logger.warning("ADS API key not provided; skipping ADS source.")
        return []

    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {NASA_API_KEY}"}
    params = {"q": query, "fl": "title,bibcode,abstract,author,year,identifier", "rows": max_results, "start": start}
    papers: List[Dict[str, Any]] = []

    try:
        logger.info(f"Querying ADS: q='{query}', rows={max_results}, start={start}")
        resp = SESSION.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []

        for idx, doc in enumerate(docs[:max_results]):
            bibcode = doc.get("bibcode") or f"ADS_{start + idx}"
            title_item = doc.get("title")
            title = title_item[0] if isinstance(title_item, list) and title_item else (title_item or "Unknown Title")
            summary_item = doc.get("abstract")
            summary = summary_item[0] if isinstance(summary_item, list) and summary_item else (summary_item or "")
            authors = doc.get("author") or []
            year_item = doc.get("year")
            year = year_item[0] if isinstance(year_item, list) and year_item else (year_item or "")

            paper = {
                "id": str(bibcode),
                "source": "ADS",
                "title": title,
                "summary": summary,
                "authors": authors,
                "year": year,
                "pdf_link": None,  # ADS often requires extra resolution to get PDFs
                "metadata": {k: v for k, v in doc.items() if k not in {"id", "title", "abstract", "bibcode"}}
            }
            papers.append(paper)

    except requests.exceptions.RequestException as e:
        logger.error(f"ADS request error: {e}")
    except ValueError as e:
        logger.error(f"ADS JSON decode error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in fetch_from_ads: {e}")

    return papers

# -------------------------
# Unified interface
# -------------------------
def search_nasa_papers(
    query: str,
    max_results: int = MAX_PAPERS_FETCH,
    source: str = "NTRS",
    start: int = 0,
    mission: Optional[str] = None,
    organism: Optional[str] = None,
    health_domain: Optional[str] = None,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Unified search for NASA papers with advanced filtering and caching.
    """
    if not query:
        return []

    # Append filters to query string
    filter_terms = [term for term in [mission, organism, health_domain] if term]
    if filter_terms:
        full_query = f"{query} AND ({' AND '.join(filter_terms)})"
    else:
        full_query = query

    # Caching logic
    search_params = {"query": full_query, "max_results": max_results, "source": source, "start": start}
    cache_key = get_cache_key(source.lower(), search_params)
    if use_cache:
        cached_data = read_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

    src = (source or "NTRS").upper()
    if src == "NTRS":
        results = fetch_from_ntrs(full_query, max_results=max_results, start=start)
    elif src == "ADS":
        results = fetch_from_ads(full_query, max_results=max_results, start=start)
    else:
        logger.error("Invalid source. Use 'NTRS' or 'ADS'.")
        results = []

    if use_cache:
        write_to_cache(cache_key, results)

    return results

def search_nasa_papers_paginated(query: str, total_limit: int = 100, **kwargs):
    """
    A generator that fetches papers page by page up to a total limit.
    """
    fetched_count = 0
    page_size = min(50, total_limit) # Fetch in pages of 50 or less

    while fetched_count < total_limit:
        remaining = total_limit - fetched_count
        current_batch_size = min(page_size, remaining)
        
        page_results = search_nasa_papers(query, max_results=current_batch_size, start=fetched_count, **kwargs)
        
        if not page_results:
            logger.info("No more results found from API.")
            break

        for paper in page_results:
            yield paper
            fetched_count += 1
            if fetched_count >= total_limit:
                break
        
        # If the API returned fewer results than we asked for, it's the last page.
        if len(page_results) < current_batch_size:
            break

# Alias for backward compatibility and clarity
get_plain_papers_for_pipeline = search_nasa_papers

# -------------------------
# CLI test block
# -------------------------
if __name__ == "__main__":
    try:
        query = input("Enter search query for NASA papers: ").strip()
        source = input("Choose source (NTRS/ADS) [default NTRS]: ").strip() or "NTRS"
        max_results = 5

        logger.info(f"Searching {source} for: {query} (max_results={max_results})")
        papers = search_nasa_papers(query, max_results=max_results, source=source, use_cache=False)

        print(f"\nFound {len(papers)} papers. Showing brief preview:\n")
        for i, p in enumerate(papers, start=1):
            print(f"--- Paper {i} ---")
            print(f"ID    : {p.get('id')}")
            print(f"Title : {p.get('title')}")
            print(f"PDF   : {p.get('pdf_link')}")
            print(f"Summary: {p.get('summary', 'N/A')[:300]}...\n")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.exception(f"Test run failed: {e}")
