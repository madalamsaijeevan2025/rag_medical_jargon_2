import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import structlog

from config import get_settings
import llm_provider
from logging_config import configure_logging
from health import router as health_router
from retriever import index_page, search
from vlm import check_range, explain_detailed, extract
from dictionary_service import record_feedback, get_feedback_summary
from document_store import document_store
from cleanup_job import start_scheduler
from term_indexer import index_terms_for_document, load_term_index
import seed_medical_db
import asyncio
from concurrent.futures import ThreadPoolExecutor


settings = get_settings()
configure_logging()
logger = structlog.get_logger()

# Thread pool for background term indexing
TERM_INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="term-index")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Auto-seed medical dictionary if missing
    from pathlib import Path
    if not Path(settings.dict_db_path).exists():
        logger.info("dict.seeding", path=settings.dict_db_path)
        seed_medical_db.init_db()
    
    scheduler = start_scheduler()
    logger.info("app.startup")
    yield
    # Shutdown
    scheduler.shutdown()
    TERM_INDEX_EXECUTOR.shutdown(wait=True)
    logger.info("app.shutdown")


app = FastAPI(title="Document Intelligence Service", lifespan=lifespan)

# Mount health router
app.include_router(health_router)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


OUTPUT_DIR = Path(settings.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")


@app.post("/upload")
async def upload_and_convert_pdf(
    request: Request,
    file: UploadFile = File(...),
    dpi: int = Query(default=300, ge=settings.min_dpi, le=settings.max_dpi),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    pdf_bytes = await file.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error("pdf.parse_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF stream: {e}")

    if len(doc) > settings.max_pages:
        raise HTTPException(
            status_code=413,
            detail=f"PDF has {len(doc)} pages; the limit is {settings.max_pages}.",
        )

    doc_id = Path(file.filename).stem
    saved_pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        image_name = f"{doc_id}_page_{page_num + 1}.png"
        image_path = OUTPUT_DIR / image_name
        pix.save(str(image_path))

        try:
            # Run in a worker thread: the embed call can take many seconds
            # (cloud API or VLM), and blocking the event loop here would
            # freeze ALL other endpoints (the app's status polls timed out).
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                index_page,
                doc_id, page_num + 1, page.get_text(), str(image_path.resolve()),
            )
        except Exception as e:
            logger.error("indexing.failed", doc_id=doc_id, page=page_num + 1, error=str(e))
            raise HTTPException(
                status_code=503,
                detail=f"Indexing failed. Is Ollama running and is nomic-embed-text pulled? ({e})",
            )

        saved_pages.append({
            "page": page_num + 1,
            "filename": image_name,
            "resolution": f"{pix.width}x{pix.height}px",
            "path": str(image_path.resolve()),
            "url": f"{str(request.base_url)}static/{image_name}",
        })

    doc.close()

    # Start term indexing in background
    if len(saved_pages) <= settings.max_pages:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(TERM_INDEX_EXECUTOR, index_terms_for_document, doc_id)
        logger.info("upload.term_indexing_started", doc_id=doc_id)
    else:
        logger.info("upload.term_indexing_skipped", doc_id=doc_id, pages=len(saved_pages))

    logger.info("upload.success", doc_id=doc_id, pages=len(saved_pages), dpi=dpi)
    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file.filename,
        "total_pages": len(saved_pages),
        "dpi": dpi,
        "pages": saved_pages,
    }


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    k: int = 2


@app.post("/query")
async def query_document(req: QueryRequest):
    t0 = time.time()
    pages = search(req.question, req.doc_id, k=min(req.k, 3))
    if not pages:
        raise HTTPException(status_code=404, detail="No indexed pages for this document.")
    t1 = time.time()

    try:
        result = extract(req.question, pages)
        t2 = time.time()
        explained = explain_detailed(result.findings)
        what_it_is = explained["sentences"]
        definition_sources = explained["sources"]
        definition_qualities = explained["qualities"]
        t3 = time.time()
    except Exception as e:
        logger.error("vlm.failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"VLM call failed: {e}")

    logger.info("query.timings", search=f"{t1-t0:.2f}s", extract=f"{t2-t1:.2f}s", explain=f"{t3-t2:.2f}s")

    rows = [
        {**f.model_dump(),
         "status": check_range(f.value, f.reference_range),
         "what_it_measures": what_it_is.get(f.name, ""),
         "definition_source": definition_sources.get(f.name, ""),
         "definition_quality": definition_qualities.get(f.name, 0.0)}
        for f in result.findings
    ]
    return {
        "pages_used": [p["page"] for p in pages],
        "result": {"answer": result.answer, "findings": rows},
    }


# Document lifecycle endpoints
@app.get("/documents")
async def list_documents():
    docs = document_store.list_docs()
    return [{"doc_id": d.doc_id, "page_count": d.page_count, "created_at": d.created_at, "size_bytes": d.size_bytes} for d in docs]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    # Delete images
    for img_path in OUTPUT_DIR.glob(f"{doc_id}_page_*.png"):
        try:
            img_path.unlink()
            logger.info("document.image_deleted", file=img_path.name)
        except Exception as e:
            logger.warning("document.image_delete_failed", file=img_path.name, error=str(e))

    success = document_store.delete_doc(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Also delete term index
    index_path = Path("./term_indexes") / f"{doc_id}.json"
    if index_path.exists():
        index_path.unlink()
    
    logger.info("document.deleted", doc_id=doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# Explanation feedback endpoints (Goal #6)
class FeedbackRequest(BaseModel):
    term: str
    rating: str  # one of: poor, fair, good
    comment: str | None = None


@app.post("/feedback/explanation")
async def submit_explanation_feedback(req: FeedbackRequest):
    """Record user feedback on a term explanation; 'poor' flags it for re-explanation."""
    if req.rating not in ("poor", "fair", "good"):
        raise HTTPException(status_code=422, detail="rating must be one of: poor, fair, good")
    if not req.term.strip():
        raise HTTPException(status_code=422, detail="term must not be empty")
    success = record_feedback(req.term.strip(), req.rating, req.comment)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to record feedback")
    logger.info("feedback.recorded", term=req.term, rating=req.rating)
    return {"status": "recorded", "term": req.term, "rating": req.rating}


@app.get("/explanations/quality")
async def explanation_quality_summary():
    """Aggregate explanation feedback stats (satisfaction score, flagged terms)."""
    return get_feedback_summary()


# --- LLM provider configuration (local Ollama or cloud API key) ---
class LLMConfigRequest(BaseModel):
    provider: str                      # "local" (Ollama) or "api" (OpenAI-compatible)
    api_key: str | None = None         # required (once) when switching to "api"
    base_url: str | None = None        # e.g. https://api.openai.com/v1
    model: str | None = None           # vision-capable chat model
    embed_model: str | None = None     # embedding model


@app.get("/llm/status")
async def llm_status():
    """Current AI backend: local Ollama or cloud API (key masked)."""
    return llm_provider.status()


@app.post("/llm/config")
async def set_llm_config(req: LLMConfigRequest):
    """Switch between local Ollama and a cloud (OpenAI-compatible) API at runtime.

    Then upload/query/term-indexing all work without Ollama installed. The API
    key is stored only in process memory and is never returned by any endpoint.
    """
    try:
        info = llm_provider.configure(
            req.provider,
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model,
            embed_model=req.embed_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "configured", **info}


# Term index endpoints
@app.get("/terms/{doc_id}")
async def get_term_index(doc_id: str):
    """Get the medical term index for a document."""
    index = load_term_index(doc_id)
    if not index:
        raise HTTPException(status_code=404, detail="Term index not found. It may still be generating or the document was not processed.")
    return index


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)