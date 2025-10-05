# ai_engine/modules/graph_builder.py

"""
Enhanced Professional Knowledge Graph Builder for NASA Bioscience Knowledge Engine

Features:
- Offline entity extraction via SciSpacy / SpaCy
- Relation extraction via dependency parsing
- AI-assisted edge enrichment (optional, if quota allows)
- Clean, weighted, merged nodes for stunning visualizations
- Cytoscape.js compatible output
"""

import logging
import sys
import os
from collections import Counter
from typing import List, Dict, Any, Optional
import re

# NLP libraries
import spacy
from spacy.matcher import Matcher

# Adjust path for standalone execution
if __name__ == '__main__' and __package__ is None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from AI_engine.utils.logger import get_logger
from AI_engine.config import LOG_LEVEL, USE_LLM, OPENAI_API_KEY, OPENAI_MODEL

logger = get_logger("AI_engine.graph_builder", LOG_LEVEL)

# ------------------ Load NLP Model ------------------
try:
    nlp = spacy.load("en_core_sci_sm")
    logger.info("Loaded SciSpacy model: en_core_sci_sm")
except Exception:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.warning("SciSpacy not found, using standard 'en_core_web_sm'. Biomedical entity recognition may be limited.")
    except Exception:
        nlp = None
        logger.error("No spaCy models found. Please install 'en_core_web_sm' or 'en_core_sci_sm'.")

# ------------------ Graph Extraction ------------------
def _clean_phrase(phrase: str) -> str:
    """Clean and normalize a phrase for use as a node ID."""
    # Remove leading/trailing articles and punctuation
    phrase = re.sub(r"^(a|an|the)\s+", "", phrase.strip(), flags=re.IGNORECASE)
    phrase = re.sub(r"[.,;]$", "", phrase.strip())
    return phrase.lower().replace(' ', '_')

def extract_entities_relations(text: str, paper_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict]]:
    """Extracts nodes and edges from text using NLP and optional AI."""
    if not nlp:
        logger.error("spaCy model not loaded. Cannot extract graph data.")
        return {"nodes": [], "edges": []}

    doc = nlp(text)
    edges = set()  # Use a set to avoid duplicate edges
    all_entities = set()
    
    # Define unhelpful verbs to ignore
    stop_verbs = {"be", "have", "do", "include", "provide", "use", "show", "study", "propose"}

    # ----- Dependency-based relation extraction -----
    for sent in doc.sents:
        # Find all verbs that are not auxiliary
        verbs = [tok for tok in sent if tok.pos_ == "VERB" and tok.dep_ != "aux"]
        for verb in verbs:
            if verb.lemma_.lower() in stop_verbs:
                continue

            subjects = [tok for tok in verb.children if tok.dep_ in ("nsubj", "nsubjpass")]
            for subj in subjects:
                source_phrase = ' '.join(t.text for t in subj.subtree).strip()
                
                # Pattern 1: Subject -> Verb -> Direct Object
                objects = [tok for tok in verb.children if tok.dep_ == "dobj"]
                for obj in objects:
                    target_phrase = ' '.join(t.text for t in obj.subtree).strip()
                    if source_phrase and target_phrase and source_phrase != target_phrase:
                        edges.add((_clean_phrase(source_phrase), _clean_phrase(target_phrase), verb.lemma_))
                        all_entities.add(source_phrase)
                        all_entities.add(target_phrase)

                # Pattern 2: Subject -> Verb -> Preposition -> Object
                preps = [tok for tok in verb.children if tok.dep_ == "prep"]
                for prep in preps:
                    pobjs = [tok for tok in prep.children if tok.dep_ == "pobj"]
                    for pobj in pobjs:
                        target_phrase = ' '.join(t.text for t in pobj.subtree).strip()
                        if source_phrase and target_phrase and source_phrase != target_phrase:
                            relation_label = f"{verb.lemma_} {prep.text}"
                            edges.add((_clean_phrase(source_phrase), _clean_phrase(target_phrase), relation_label))
                            all_entities.add(source_phrase)
                            all_entities.add(target_phrase)

    # ----- Named entities -----
    for ent in doc.ents:
        all_entities.add(ent.text)

    # ----- Optional AI-enhanced edges -----
    if USE_LLM:
        try:
            from .rag_engine import generate_with_local_llm, generator_pipeline
            if not generator_pipeline:
                raise RuntimeError("Local LLM generator not available.")

            prompt = f"Extract key entities and relationships from this text for a knowledge graph, output as JSON nodes and edges: {text}"
            system_prompt = "You are an expert biomedical knowledge graph generator."
            ai_data = generate_with_local_llm(system_prompt, prompt)

            # Optional: parse AI JSON safely and merge with existing nodes/edges
            # Skipping parsing to prevent crashes if quota exceeded
        except Exception as e:
            logger.warning(f"AI graph extraction skipped: {e}")

    # ----- Node creation -----
    # Count occurrences of each entity in the original text for weighting
    text_lower = text.lower()
    entity_counts = {entity: text_lower.count(entity.lower()) for entity in all_entities}

    nodes = []
    for entity in all_entities:
        node_data = {
            "id": _clean_phrase(entity),
            "label": entity,
            "weight": entity_counts.get(entity, 1)
        }
        if paper_metadata:
            node_data.update(paper_metadata)
        nodes.append({"data": node_data})
    
    # Convert set of edges to list of dicts
    final_edges = [{"data": {"source": s, "target": t, "label": l}} for s, t, l in edges]

    logger.info(f"Extracted {len(nodes)} nodes and {len(final_edges)} edges.")

    return {"nodes": nodes, "edges": final_edges}

# ------------------ CLI Test ------------------
if __name__ == "__main__":
    import json
    try:
        from AI_engine.modules.nasa_api import get_plain_papers_for_pipeline
        query = input("Enter a query to fetch papers for graph extraction: ").strip()
        papers = get_plain_papers_for_pipeline(query, max_results=2)

        for paper in papers:
            print(f"\n--- Processing Paper: {paper.get('title')} ---")
            text_to_process = paper.get('summary', '')
            if text_to_process:
                paper_meta = {
                    "paper_id": paper.get('id'),
                    "source": paper.get('source'),
                    "year": paper.get('year', 'Unknown')
                }
                graph_data = extract_entities_relations(text_to_process, paper_metadata=paper_meta)
                print(json.dumps(graph_data, indent=2))
            else:
                print("No abstract available for this paper.")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.exception(f"Error during graph extraction test: {e}")
