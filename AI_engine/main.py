# ai_engine/main.py

"""
NASA Bioscience Knowledge Engine - Main Pipeline

Responsibilities:
- Orchestrates the full AI pipeline: fetch, process, embed, retrieve, generate answers, summarize, and build knowledge graph.
- Can be used as CLI or backend API.
- Modular structure: calls RAG engine, summarizer, and graph builder modules.
"""

import logging
import csv
from datetime import datetime
import textwrap
from typing import Dict
from .config import LOG_LEVEL, OPENAI_API_KEY, USE_LLM, MAX_PAPERS_FETCH, DATA_DIR
from .modules.rag_engine import run_rag_pipeline
from .modules.summarizer import summarize
from .modules.graph_builder import extract_entities_relations
from .modules.nasa_api import search_nasa_papers
from .modules.data_processor import process_papers
from .modules.embedding_store import build_store
from .utils.logger import get_logger

logger = get_logger("AI_engine.main", LOG_LEVEL)

# --- Export Functionality ---
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("`reportlab` not found. PDF export will be disabled. Run `pip install reportlab`.")


# ------------------ PIPELINE FUNCTION ------------------
def run_full_pipeline(query: str, fetch_new_papers: bool = False) -> Dict:
    """
    Executes the full AI pipeline for a given query.

    Args:
        query (str): User search query or research topic.
        fetch_new_papers (bool): If True, fetches new papers and rebuilds the vector store.

    Returns:
        dict: A comprehensive dictionary with all pipeline outputs.
    """
    logger.info(f"--- Running AI Engine for query: '{query}' ---")

    # Optional: Fetch new papers and build a temporary, in-memory knowledge base
    if fetch_new_papers:
        logger.info(f"Fetching {MAX_PAPERS_FETCH} new papers for '{query}'...")
        raw_papers = search_nasa_papers(query, max_results=MAX_PAPERS_FETCH, use_cache=True)
        processed_chunks = process_papers(raw_papers)
        # Build a temporary store for this session, don't save to disk
        temp_index, temp_chunks = build_store(processed_chunks, save_to_disk=False)
        rag_kwargs = {"index": temp_index, "chunks": temp_chunks}
    else:
        # Use the existing, persistent knowledge base
        rag_kwargs = {}

    # 1. Run RAG pipeline to get answer and retrieved context
    rag_results = run_rag_pipeline(query, **rag_kwargs)
    retrieved_chunks = rag_results.get('retrieved_chunks', [])

    summary_results = {}
    graph_data = {"nodes": [], "edges": []}

    if retrieved_chunks:
        # 2. Combine text from retrieved chunks for summarization and graph building
        combined_text = " ".join([chunk.get('chunk', '') for chunk in retrieved_chunks])
        
        # Summarization
        logger.info("Generating summary from retrieved context...")
        summary_results = summarize(combined_text)

        # Knowledge Graph
        logger.info("Building knowledge graph from retrieved context...")
        graph_data = extract_entities_relations(combined_text)
    else:
        logger.warning("No chunks retrieved; skipping summarization and graph building.")

    # 3. Assemble the final output
    final_output = {
        "query": query,
        "rag_answer": rag_results.get('answer'),
        "summary_of_context": summary_results,
        "knowledge_graph": graph_data,
        "retrieved_chunks": retrieved_chunks
    }

    return final_output


# ------------------ EXPORT FUNCTIONS ------------------
def export_to_pdf(result: Dict, filename: str):
    """Exports the results to a PDF file."""
    if not REPORTLAB_AVAILABLE:
        logger.error("Cannot export to PDF, `reportlab` is not installed.")
        return

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='Query', fontSize=14, leading=16, spaceAfter=12, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Header', fontSize=12, leading=14, spaceAfter=6, fontName='Helvetica-Bold'))
    
    story = []

    # Title
    story.append(Paragraph(f"Query: {result['query']}", styles['Query']))

    # RAG Answer
    story.append(Paragraph("RAG Answer", styles['Header']))
    rag_answer = result.get('rag_answer', 'N/A').replace('\n', '<br/>')
    story.append(Paragraph(rag_answer, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Summary
    story.append(Paragraph("Summary of Retrieved Context", styles['Header']))
    summary = result.get('summary_of_context', {}).get('summary', 'N/A').replace('\n', '<br/>')
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Sources
    story.append(Paragraph("Retrieved Sources", styles['Header']))
    for i, chunk in enumerate(result.get('retrieved_chunks', [])[:5], 1):
        source_text = f"<b>[{i}] {chunk.get('title', 'N/A')} (ID: {chunk.get('paper_id', 'N/A')})</b><br/>"
        source_text += textwrap.shorten(chunk.get('chunk', ''), width=120, placeholder="...")
        story.append(Paragraph(source_text, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    logger.info(f"Results successfully exported to {filename}")

def export_to_csv(result: Dict, filename: str):
    """Exports the key results to a CSV file."""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Query', 'RAG Answer', 'Summary', 'Source 1 Title', 'Source 1 Snippet', 'Source 2 Title', 'Source 2 Snippet'])
            
            # Data
            rag_answer = result.get('rag_answer', 'N/A')
            summary = result.get('summary_of_context', {}).get('summary', 'N/A')
            
            sources = result.get('retrieved_chunks', [])
            s1_title = sources[0].get('title', '') if len(sources) > 0 else ''
            s1_chunk = sources[0].get('chunk', '') if len(sources) > 0 else ''
            s2_title = sources[1].get('title', '') if len(sources) > 1 else ''
            s2_chunk = sources[1].get('chunk', '') if len(sources) > 1 else ''

            writer.writerow([
                result['query'], rag_answer, summary, 
                s1_title, s1_chunk, s2_title, s2_chunk
            ])
        logger.info(f"Results successfully exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")


# ------------------ CLI FUNCTION ------------------
def run_cli():
    """Run the engine in CLI mode."""
    print("--- NASA Bioscience Knowledge Engine ---")
    user_query = input("Enter your query (e.g., 'dark energy', 'space nutrition'): ").strip()
    if not user_query:
        logger.warning("No query entered. Exiting.")
        return
    
    if not OPENAI_API_KEY and USE_LLM:
        logger.error("OpenAI API key is not set, but USE_LLM is True.")
        logger.error("Please set your OPENAI_API_KEY in the .env file or disable USE_LLM in config.py.")
        return

    fetch_new = input("Fetch new papers for this query? (y/N): ").strip().lower()
    should_fetch = fetch_new == 'y'

    result = run_full_pipeline(user_query, fetch_new_papers=should_fetch)

    # Display results in CLI
    print("\n" + "="*80)
    print(f"QUERY: {result.get('query')}")
    print("="*80 + "\n")

    print("--- RAG Answer ---")
    print(result.get('rag_answer', 'N/A'), "\n")

    print("--- Summary of Retrieved Context ---")
    print(result.get('summary_of_context', {}).get('summary', 'N/A'), "\n")

    print(f"--- Knowledge Graph ---")
    print(f"Extracted {len(result['knowledge_graph']['nodes'])} nodes and {len(result['knowledge_graph']['edges'])} edges.\n")

    print("--- Retrieved Sources (Top 3) ---")
    for i, chunk in enumerate(result.get('retrieved_chunks', [])[:3], 1):
        print(f"[{i}] {chunk.get('title', 'N/A')}: {chunk.get('chunk', '')[:150]}...")
    print("\n" + "="*80)

    # Ask for export
    export_choice = input("Export results? (pdf/csv/No): ").strip().lower()
    if export_choice in ['pdf', 'csv']:
        export_dir = DATA_DIR / "exports"
        export_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = export_dir / f"nasa_engine_results_{timestamp}.{export_choice}"
        if export_choice == 'pdf':
            export_to_pdf(result, str(filename))
        elif export_choice == 'csv':
            export_to_csv(result, str(filename))
    else:
        logger.info("Skipping export.")


# ------------------ API FUNCTION ------------------
def run_api():
    """Return a FastAPI instance to expose the engine as an API."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="NASA Bioscience Knowledge Engine API")

    class QueryRequest(BaseModel):
        query: str
        fetch_new_papers: bool = False

    @app.post("/run_pipeline")
    def pipeline_endpoint(request: QueryRequest):
        return run_full_pipeline(request.query, fetch_new_papers=request.fetch_new_papers)

    return app


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    run_cli()
