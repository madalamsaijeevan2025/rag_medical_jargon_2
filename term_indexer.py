import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import structlog

from config import get_settings
from retriever import search
from vlm import (extract, explain_detailed, SECTION_HINTS, LAB_TERM_HINTS,
                 PAGE_LAYOUT_GUIDANCE)
from document_store import document_store
from dictionary_service import lookup_terms

logger = structlog.get_logger()
settings = get_settings()

BATCH_PAGES = 4
MAX_PAGES = 50


def _page_hash(image_path: str) -> str:
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _build_extract_prompt(page_nums: List[int]) -> str:
    return (
        f"Pages are in order: {page_nums}.\n"
        "TASK: Extract every clinically meaningful term visible on these pages.\n"
        "INCLUDE: lab test names, abbreviations, radiology and pathology findings, "
        "conditions/diagnoses, procedures, anatomy terms, and medications if printed.\n"
        f"HIGH-PRIORITY sections: {', '.join(SECTION_HINTS)}.\n"
        f"ANCHOR TERMS (do not miss these if they are visible): {', '.join(LAB_TERM_HINTS)}.\n"
        + PAGE_LAYOUT_GUIDANCE + "\n"
        "For each term, include: name, value (if present), unit (if present), "
        "reference_range (if present), page number.\n"
        "Return JSON matching the Extraction schema. If no terms found, return empty "
        "findings array. Do not give medical advice or diagnoses."
    )


def index_terms_for_document(doc_id: str) -> Dict[str, Any]:
    """Extract and explain all medical terms from a document."""
    meta = document_store.get_meta(doc_id)
    if not meta:
        raise ValueError(f"Document {doc_id} not found")
    
    total_pages = min(meta.page_count, MAX_PAGES)
    if total_pages == 0:
        raise ValueError("Document has no pages")
    
    page_numbers = list(range(1, total_pages + 1))
    batches = [page_numbers[i:i + BATCH_PAGES] for i in range(0, len(page_numbers), BATCH_PAGES)]
    
    all_findings = []
    
    for batch_idx, batch_pages in enumerate(batches):
        logger.info("term_indexer.batch_start", doc_id=doc_id, batch=batch_idx + 1, total=len(batches), pages=batch_pages)
        
        page_metas = []
        for p in batch_pages:
            results = search(f"page {p}", doc_id, k=1)
            if results and results[0].get("page") == p:
                page_metas.append(results[0])
        
        if not page_metas:
            logger.warning("term_indexer.no_pages_found", batch_pages=batch_pages)
            continue
        
        question = _build_extract_prompt(batch_pages)
        pages_for_vlm = [{"page": m["page"], "image_path": m["image_path"]} for m in page_metas]
        
        try:
            extraction = extract(question, pages_for_vlm)
            all_findings.extend(extraction.findings)
            logger.info("term_indexer.batch_extracted", batch=batch_idx + 1, findings=len(extraction.findings))
        except Exception as e:
            logger.error("term_indexer.batch_failed", batch=batch_idx + 1, error=str(e))
            continue
    
    # Deduplicate by term name
    seen = set()
    unique_findings = []
    for f in all_findings:
        key = f.name.lower()
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)
    
    logger.info("term_indexer.deduped", total=len(all_findings), unique=len(unique_findings))
    
    # Get explanations (hybrid dict + VLM with source + quality tracking)
    explained = explain_detailed(unique_findings)
    explanations = explained["sentences"]
    explanation_sources = explained["sources"]
    explanation_qualities = explained["qualities"]
    
    # Build term index
    term_index = {}
    for f in unique_findings:
        term_index[f.name] = {
            "name": f.name,
            "definition": explanations.get(f.name, ""),
            "definition_source": explanation_sources.get(f.name, ""),
            "definition_quality": explanation_qualities.get(f.name, 0.0),
            "pages": [f.page],
            "example_value": f.value if f.value else None,
            "unit": f.unit if f.unit else None,
            "reference_range": f.reference_range if f.reference_range else None,
        }
    
    # Merge pages for duplicate terms
    for f in all_findings:
        if f.name in term_index and f.page not in term_index[f.name]["pages"]:
            term_index[f.name]["pages"].append(f.page)
    
    result = {
        "doc_id": doc_id,
        "total_pages": total_pages,
        "terms_found": len(term_index),
        "terms": term_index,
        "indexed_at": time.time(),
    }
    
    # Save to file
    index_dir = Path("./term_indexes")
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{doc_id}.json"
    import json
    index_path.write_text(json.dumps(result, indent=2))
    
    logger.info("term_indexer.completed", doc_id=doc_id, terms=len(term_index))
    return result


def load_term_index(doc_id: str) -> Dict[str, Any]:
    index_path = Path("./term_indexes") / f"{doc_id}.json"
    if not index_path.exists():
        return None
    import json
    return json.loads(index_path.read_text())