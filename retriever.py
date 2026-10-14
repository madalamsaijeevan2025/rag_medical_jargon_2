import chromadb
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import get_settings
import llm_provider
from cache import embedding_cache
from document_store import document_store
import structlog

logger = structlog.get_logger()

settings = get_settings()

client = chromadb.PersistentClient(path=settings.db_path)
col = client.get_or_create_collection("pages", metadata={"hnsw:space": "cosine"})


def _is_transient(exc: Exception) -> bool:
    # Treat network / timeout errors as transient
    return isinstance(exc, (ollama.ResponseError, llm_provider.ProviderError,
                            ConnectionError, TimeoutError))


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(settings.ollama_max_retries),
    retry=retry_if_exception_type((ollama.ResponseError, llm_provider.ProviderError,
                                   ConnectionError, TimeoutError)),
    reraise=True,
)
def _embed_with_retry(texts, kind):
    # kind: "search_document" or "search_query"
    model = llm_provider.embed_model()
    # Try cache for single text
    if len(texts) == 1:
        cached = embedding_cache.get(model, texts[0])
        if cached is not None:
            logger.debug("embed.cache_hit", model=model)
            return cached

    if llm_provider.status()["provider"] == "api":
        # Cloud fallback: OpenAI-compatible /embeddings (no ollama options).
        r = llm_provider.embed(model=model, input=[f"{kind}: {t[:6000]}" for t in texts])
    else:
        r = ollama.embed(
            model=model,
            input=[f"{kind}: {t[:6000]}" for t in texts],
            options={"num_ctx": 8192},
        )
    embeddings = r["embeddings"]

    if len(texts) == 1:
        embedding_cache.set(model, texts[0], embeddings)
        logger.debug("embed.cache_set", model=model)

    return embeddings


def index_page(doc_id, page_no, text, image_path):
    embeddings = _embed_with_retry([text], "search_document")
    col.upsert(
        ids=[f"{doc_id}:{page_no}"],
        documents=[text],
        embeddings=embeddings,
        metadatas=[{"doc_id": doc_id, "page": page_no, "image_path": image_path}],
    )
    # Update document metadata
    document_store.upsert_page(doc_id, page_no, text, image_path)


def search(query, doc_id, k=3):
    embeddings = _embed_with_retry([query], "search_query")
    r = col.query(query_embeddings=embeddings, n_results=k, where={"doc_id": doc_id})
    if not r["metadatas"] or not r["metadatas"][0]:
        return []
    return [{**m, "distance": d} for m, d in zip(r["metadatas"][0], r["distances"][0])]