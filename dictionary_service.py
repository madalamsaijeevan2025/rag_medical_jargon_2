import sqlite3
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from config import get_settings
from cache import explain_cache

_settings = get_settings()
_DB_PATH = _settings.dict_db_path
_local = threading.local()


def _get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def _ensure_db():
    """Create table/index if missing (idempotent)."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medical_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            definition TEXT NOT NULL,
            clinical_significance TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_term_lower 
        ON medical_terms (LOWER(term))
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS explanation_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT NOT NULL,
            rating TEXT NOT NULL CHECK (rating IN ('good','fair','poor')),
            comment TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_term
        ON explanation_feedback (term)
    """)
    conn.commit()


_ensure_db()


def lookup_terms(extracted_terms: list[str]) -> dict:
    """
    Batch lookup against local SQLite.
    Returns: {"found": {term: {...}}, "missing": [terms]}
    """
    if not extracted_terms:
        return {"found": {}, "missing": []}

    conn = _get_conn()
    placeholders = ",".join(["?"] * len(extracted_terms))
    query = f"""
        SELECT term, category, definition, clinical_significance 
        FROM medical_terms 
        WHERE LOWER(term) IN ({','.join(['LOWER(?)'] * len(extracted_terms))})
    """
    rows = conn.execute(query, extracted_terms).fetchall()

    found = {}
    found_lower = set()
    for row in rows:
        term = row["term"]
        found[term] = {
            "category": row["category"],
            "definition": row["definition"],
            "significance": row["clinical_significance"],
            "source": "Local Medical DB",
        }
        found_lower.add(term.lower())

    missing = [t for t in extracted_terms if t.lower() not in found_lower]
    return {"found": found, "missing": missing}


def add_term(term: str, category: str, definition: str, significance: str = "") -> bool:
    """Insert or update a term. Returns True if inserted/updated."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO medical_terms (term, category, definition, clinical_significance)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                category=excluded.category,
                definition=excluded.definition,
                clinical_significance=excluded.clinical_significance
            """,
            (term, category, definition, significance),
        )
        conn.commit()
        # Invalidate explain cache since new term available
        explain_cache._backend.clear_prefix("explain:")
        return True
    except Exception:
        return False


def bulk_add(terms: list[tuple[str, str, str, str]]) -> int:
    """Bulk insert. Returns count of inserted/updated."""
    conn = _get_conn()
    cur = conn.executemany(
        """
        INSERT INTO medical_terms (term, category, definition, clinical_significance)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(term) DO UPDATE SET
            category=excluded.category,
            definition=excluded.definition,
            clinical_significance=excluded.clinical_significance
        """,
        terms,
    )
    conn.commit()
    if cur.rowcount:
        explain_cache._backend.clear_prefix("explain:")
    return cur.rowcount


def get_all_terms() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT term, category, definition, clinical_significance FROM medical_terms ORDER BY term").fetchall()
    return [dict(r) for r in rows]


# --- Explanation feedback loop (Goal #6) ---
FEEDBACK_DECAY_DAYS = 7
_VALID_RATINGS = ("good", "fair", "poor")


def record_feedback(term: str, rating: str, comment: str | None = None) -> bool:
    """Store user feedback about an explanation.

    rating must be 'good', 'fair' or 'poor'. A 'poor' rating flags the term so
    the next explanation request forces a fresh VLM definition (re-explain).
    """
    if rating not in _VALID_RATINGS:
        return False
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO explanation_feedback (term, rating, comment, created_at) "
            "VALUES (?, ?, ?, ?)",
            (term, rating, comment, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True
    except Exception as e:
        logger_msg = f"feedback.record_failed: {e}"
        print(logger_msg)  # no structlog dependency here; keep module standalone
        return False


def has_recent_negative_feedback(term: str) -> bool:
    """True when the term got a 'poor' rating within FEEDBACK_DECAY_DAYS days
    with no later 'good' rating to clear the flag."""
    conn = _get_conn()
    cutoff = (datetime.utcnow() - timedelta(days=FEEDBACK_DECAY_DAYS)).isoformat()
    row = conn.execute(
        """
        SELECT rating, created_at FROM explanation_feedback
        WHERE LOWER(term) = LOWER(?)
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (term,),
    ).fetchone()
    if row is None:
        return False
    return row["rating"] == "poor" and str(row["created_at"]) >= cutoff


def get_feedback_summary() -> dict:
    """Aggregate feedback stats: counts, satisfaction score (out of 5), flagged terms."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) AS n FROM explanation_feedback").fetchone()["n"]
    rows = conn.execute(
        "SELECT rating, COUNT(*) AS n FROM explanation_feedback GROUP BY rating"
    ).fetchall()
    counts = {r["rating"]: r["n"] for r in rows}

    flagged_rows = conn.execute(
        "SELECT DISTINCT term FROM explanation_feedback WHERE rating = 'poor'"
    ).fetchall()
    flagged = sorted(r["term"] for r in flagged_rows)

    score_map = {"good": 5, "fair": 3, "poor": 1}
    total_score = sum(score_map.get(rating, 0) * n for rating, n in counts.items())
    avg = round(total_score / total, 2) if total else None

    return {
        "total_feedback": total,
        "ratings": counts,
        "avg_satisfaction_out_of_5": avg,
        "flagged_terms": flagged,
    }


if __name__ == "__main__":
    test = ["eGFR", "HbA1c", "UnknownMarkerXYZ"]
    resp = lookup_terms(test)
    print("FOUND:", list(resp["found"].keys()))
    print("MISSING:", resp["missing"])