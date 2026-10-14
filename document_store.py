import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import chromadb
from config import get_settings


@dataclass
class DocumentMeta:
    doc_id: str
    page_count: int
    created_at: str  # ISO format
    size_bytes: int


class DocumentStore:
    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.db_path)
        self._col = self._client.get_or_create_collection("pages", metadata={"hnsw:space": "cosine"})
        self._index_path = Path(settings.doc_index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, DocumentMeta] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text())
                self._meta = {k: DocumentMeta(**v) for k, v in data.items()}
            except Exception:
                self._meta = {}

    def _save_index(self):
        data = {k: asdict(v) for k, v in self._meta.items()}
        self._index_path.write_text(json.dumps(data))

    def upsert_page(self, doc_id: str, page_no: int, text: str, image_path: str):
        # Update metadata
        now = datetime.utcnow().isoformat()
        img_size = Path(image_path).stat().st_size if Path(image_path).exists() else 0
        if doc_id not in self._meta:
            self._meta[doc_id] = DocumentMeta(doc_id=doc_id, page_count=0, created_at=now, size_bytes=0)
        meta = self._meta[doc_id]
        if page_no == 1:
            # A fresh (or re-)upload starts at page 1: reset accumulated size and
            # refresh created_at so the cleanup TTL tracks the newest upload.
            meta.size_bytes = img_size
            meta.created_at = now
        else:
            meta.size_bytes += img_size
        # Count distinct pages instead of +1 per call, so re-uploading the same
        # PDF cannot inflate page_count.
        meta.page_count = max(meta.page_count, page_no)
        self._save_index()

    def list_docs(self) -> List[DocumentMeta]:
        return list(self._meta.values())

    def get_meta(self, doc_id: str) -> Optional[DocumentMeta]:
        return self._meta.get(doc_id)

    def delete_doc(self, doc_id: str) -> bool:
        if doc_id not in self._meta:
            return False
        # Delete from Chroma
        try:
            self._col.delete(where={"doc_id": doc_id})
        except Exception:
            pass
        # Remove from index
        del self._meta[doc_id]
        self._save_index()
        return True


# Singleton
document_store = DocumentStore()