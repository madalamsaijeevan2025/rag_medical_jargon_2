import io
import json
import re
import hashlib
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import ollama
from PIL import Image
from pydantic import BaseModel
from config import get_settings
import llm_provider
from cache import vlm_cache, explain_cache
from dictionary_service import lookup_terms, has_recent_negative_feedback
import structlog

logger = structlog.get_logger()

settings = get_settings()
MODEL = settings.vlm_model

# --- Explanation quality scoring (Goal #6) ---
# Text fragments that mark a definition as a failure/placeholder instead of a real
# explanation. Any definition containing one of these scores 0.
LOW_QUALITY_MARKERS = (
    "not found",
    "unknown",
    "no definition",
    "n/a",
    "this is not on",
    "i don't know",
    "cannot determine",
    "unable to",
)
# Definitions scoring below this are considered stale and get regenerated.
GOOD_QUALITY_THRESHOLD = 0.5


def score_definition(definition: str | None) -> float:
    """Heuristic quality score in [0.0, 1.0] for one explanation sentence.

    Pure and dependency-free so it can be unit tested without Ollama.
    """
    if not definition:
        return 0.0
    text = str(definition).strip()
    if not text:
        return 0.0

    low = text.lower()
    if any(marker in low for marker in LOW_QUALITY_MARKERS):
        return 0.0

    # Length heuristics: too short to be informative, too long to be a sentence.
    if len(text) < 15:
        return 0.2
    if len(text) > 400:
        return 0.6

    score = 0.6 if len(text) >= 40 else 0.5

    # A real explanation usually ends with terminal punctuation.
    if text[-1] in ".!?":
        score += 0.1

    # Avoid ALL-CAPS style output (usually labels, not explanations).
    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.7:
            return 0.3

    return min(score, 1.0)


# --- Extraction prompting guidance (Goal #2) ---
# Common section headers in medical reports; terms under these are high-priority.
SECTION_HINTS = (
    "Laboratory Results", "Results", "Complete Blood Count", "Differential",
    "Comprehensive Metabolic Panel", "Basic Metabolic Panel", "Lipid Panel",
    "Liver Function Tests", "Thyroid Panel", "Urinalysis", "Microbiology",
    "Chemistry", "Hematology", "Coagulation",
    "Vital Signs", "Physical Exam", "History of Present Illness",
    "Impressions", "Findings", "Conclusions", "Diagnosis", "Assessment",
    "Recommendations", "Technique", "Comparison", "Clinical History",
    "Pathology Report", "Biopsy Report", "Medications",
)

# Very common terms extracted from lab reports; anchors so the VLM does not miss them.
LAB_TERM_HINTS = (
    "Hemoglobin", "Hematocrit", "WBC", "RBC", "Platelets", "MCV",
    "Glucose", "HbA1c", "Creatinine", "eGFR", "BUN", "Urea",
    "Sodium", "Potassium", "Chloride", "Calcium", "Bicarbonate",
    "ALT", "AST", "ALP", "Bilirubin", "Albumin", "Total Protein",
    "Cholesterol", "Triglycerides", "LDL", "HDL",
    "TSH", "T4", "Ferritin", "Iron", "CRP", "ESR", "Troponin",
    "PT/INR", "aPTT", "Vitamin D", "Vitamin B12", "Folate",
)

# Shared layout/noise guidance appended to every extraction prompt.
PAGE_LAYOUT_GUIDANCE = (
    "PAGE READING GUIDANCE:\n"
    "- Reports are often tables or multi-column layouts. Read table rows cell by cell: "
    "each row usually pairs one test name with its value, unit, and a reference-range "
    "column. Never mix the value column with the reference-range column.\n"
    "- On two-column or boxed pages, finish the left box/column top-to-bottom before "
    "the right one; report terms from every section, not just the first.\n"
    "- Section headers (e.g. Laboratory Results, Complete Blood Count, Impressions, "
    "Findings, Diagnosis) separate content; use the section a term appears under "
    "when choosing its category.\n"
    "IGNORE as noise (never list these as terms): patient name/ID/age/sex, dates and "
    "times, accession/case/visit numbers, hospital and doctor names, signatures, "
    "addresses and phone numbers, billing/insurance codes, page numbers, letterheads, "
    "disclaimers and footer text, and standalone values, units, or dates with no "
    "test name attached."
)


# Simple circuit breaker
_cb_failures = 0
_cb_lock = Lock()
_cb_open_until = 0


def _record_success():
    global _cb_failures
    with _cb_lock:
        _cb_failures = 0


def _record_failure():
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures += 1
        if _cb_failures >= settings.ollama_cb_threshold:
            _cb_open_until = __import__("time").time() + settings.ollama_cb_reset_seconds
            logger.warning("circuit_breaker.opened", threshold=settings.ollama_cb_threshold, reset_seconds=settings.ollama_cb_reset_seconds)


def _circuit_allows() -> bool:
    with _cb_lock:
        if __import__("time").time() < _cb_open_until:
            return False
        return True


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, (ollama.ResponseError, llm_provider.ProviderError,
                            ConnectionError, TimeoutError))


class Finding(BaseModel):
    name: str
    value: str
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None
    page: int
    bbox: list[int] = []


class Extraction(BaseModel):
    answer: str
    findings: list[Finding]


def _shrink(path, max_side=1400):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _page_hash(image_path: str) -> str:
    # Hash file content for cache key stability
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(settings.ollama_max_retries),
    retry=retry_if_exception_type((ollama.ResponseError, llm_provider.ProviderError,
                                   ConnectionError, TimeoutError)),
    reraise=True,
)
def _chat_with_retry(**kwargs):
    if not _circuit_allows():
        raise RuntimeError("Circuit breaker open")
    try:
        resp = llm_provider.chat(**kwargs)
        _record_success()
        return resp
    except Exception as e:
        _record_failure()
        raise


def _get_response_text(resp) -> str:
    """Extract text from response, handling qwen3-vl thinking field."""
    content = resp["message"].get("content", "")
    if content.strip():
        return content
    thinking = resp["message"].get("thinking", "")
    if thinking.strip():
        logger.info("vlm.using_thinking_field", thinking_len=len(thinking))
        return thinking
    return ""


def _normalize_stats(resp: dict) -> dict:
    """Guarantee the Ollama timing-key shape for logging, whatever provider answered."""
    resp.setdefault("total_duration", 0)
    resp.setdefault("load_duration", 0)
    resp.setdefault("prompt_eval_duration", 0)
    resp.setdefault("eval_duration", 0)
    resp.setdefault("eval_count", 0)
    return resp


def extract_terms_from_image(image_path: str) -> list[dict]:
    """Extract medical term names and categories from a single image."""
    prompt = (
        "Extract ALL medical terms, lab test names, abbreviations, and clinical findings visible on this page. "
        "List each term with its category. "
        "Categories: 'Lab Marker', 'Radiology', 'Condition', 'Anatomy/Procedure', 'Other'. "
        "Prioritize terms under section headers such as: "
        + ", ".join(SECTION_HINTS[:8]) + ".\n"
        + PAGE_LAYOUT_GUIDANCE + "\n"
        "Only include terms that are medical/clinical. No values, just names and category. "
        "Format each line as: TERM_NAME | CATEGORY"
    )
    
    msgs = [{
        "role": "user",
        "content": prompt,
        "images": [_shrink(image_path)],
    }]
    
    try:
        r = _chat_with_retry(
            model=MODEL,
            messages=msgs,
            options={"num_ctx": 4096, "temperature": 0},
            keep_alive="10m",
        )
        
        # Get response text (handles thinking field)
        response_text = _get_response_text(r)
        if not response_text.strip():
            logger.warning("extract_terms.empty_response", image=image_path)
            return []
        
        # Parse terms directly from text
        return _parse_terms_from_text(response_text)
        
    except Exception as e:
        logger.error("extract_terms.failed", image=image_path, error=str(e))
        return []


def _parse_terms_from_text(text: str) -> list[dict]:
    """Parse terms from free-form text response (handles VLM thinking output format)."""
    import re
    valid = []
    
    # Patterns for thinking output format: "- Term Name → Category" or "Term | Category"
    # Also handles bullet points, dashes, arrows
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Skip lines that are clearly reasoning text (long sentences, start with "Wait", "Let's", "So", etc.)
        if any(line.lower().startswith(prefix) for prefix in ['wait', 'let', 'so ', 'first', 'starting', 'also', 'from the', 'also', 'wait,']):
            continue
        if len(line) > 200:  # Skip very long reasoning lines
            continue
        
        # Try patterns for "Term → Category" or "Term | Category" or "Term - Category"
        # Pattern 1: "Term → Category" (arrow) - handle unicode arrow
        arrow_match = re.match(r'^[-•*]?\s*(.+?)\s*(?:→|->)\s*(.+)$', line)
        if arrow_match:
            name = arrow_match.group(1).strip()
            category = arrow_match.group(2).strip()
            if _is_valid_term(name, category):
                valid.append({"name": name, "category": category})
            continue
        
        # Pattern 2: "Term | Category" (pipe)
        pipe_match = re.match(r'^[-•*]?\s*(.+?)\s*\|\s*(.+)$', line)
        if pipe_match:
            name = pipe_match.group(1).strip()
            category = pipe_match.group(2).strip()
            if _is_valid_term(name, category):
                valid.append({"name": name, "category": category})
            continue
        
        # Pattern 3: "Term - Category" (dash separator, but not starting dash)
        dash_match = re.match(r'^(.+?)\s+-\s+(.+)$', line)
        if dash_match and not line.startswith('-'):
            name = dash_match.group(1).strip()
            category = dash_match.group(2).strip()
            if _is_valid_term(name, category):
                valid.append({"name": name, "category": category})
            continue
    
    return valid


def _is_valid_term(name: str, category: str) -> bool:
    """Filter out non-medical terms and reasoning text."""
    name = name.strip()
    category = category.strip()
    
    # Skip if too short or too long
    if len(name) < 2 or len(name) > 80:
        return False
    if len(category) < 2 or len(category) > 40:
        return False
    
    # Valid categories (case-insensitive)
    valid_cats = {'lab marker', 'lab', 'laboratory', 'radiology', 'condition',
                  'anatomy/procedure', 'other', 'clinical finding', 'finding',
                  'abbreviation', 'drug', 'medication', 'gene', 'procedure',
                  'pathology', 'diagnosis', 'microbiology', 'chemistry',
                  'hematology', 'symptom'}
    if category.lower() not in valid_cats:
        return False
    
    # Skip reasoning text and document-noise prefixes
    skip_prefixes = ['background', 'case presentation', 'introduction', 'abstract',
                     'from the', 'also', 'wait', 'let', 'so ', 'note:', 'see',
                     'patient', 'date', 'time', 'age', 'sex', 'gender', 'name',
                     'accession', 'collected', 'received', 'reported', 'specimen',
                     'signature', 'physician', 'doctor', 'dr ', 'hospital',
                     'department', 'billing', 'insurance', 'address', 'phone',
                     'page', 'print']
    if any(name.lower().startswith(p) for p in skip_prefixes):
        return False
    
    # Skip if name looks like a sentence (too many words, ends with period)
    if name.count(' ') > 6 or name.endswith('.'):
        return False
    
    # Skip a bare value, unit, or date with no test name attached
    if re.fullmatch(r"[0-9.,%/()<>=~≤≥\s\-–]+", name):
        return False
    
    return True


def extract(question, pages):
    # Cache key: embed the effective chat model so local vs cloud results
    # never collide, and switching providers is not masked by stale pages.
    chat_model = llm_provider.chat_model()
    page_hashes = [_page_hash(p["image_path"]) for p in pages]
    cached = vlm_cache.get(chat_model, question, page_hashes)
    if cached:
        logger.info("vlm.cache_hit", question=question[:50])
        return Extraction.model_validate(cached)

    msgs = [{
        "role": "user",
        "content": (
            f"Question: {question}\n"
            f"Pages are in order: {[p['page'] for p in pages]}.\n"
            "Answer the question in 1-3 sentences in the 'answer' field, using only what is "
            "visible on the pages. If the pages do not contain the answer, say exactly: "
            "'This is not on the pages I read.' Do not give medical advice or diagnoses.\n"
            "Then list any specific test values you used in 'findings'. "
            "Copy the reference range printed next to each value into reference_range, "
            "exactly as written. Leave it empty if none is printed.\n"
            + PAGE_LAYOUT_GUIDANCE
        ),
        "images": [_shrink(p["image_path"]) for p in pages],
    }]

    r = _chat_with_retry(
        model=MODEL,
        messages=msgs,
        format=Extraction.model_json_schema(),
        options={"num_ctx": 8192, "temperature": 0},
        keep_alive="10m",
    )

    r = _normalize_stats(r)
    logger.info("vlm.timings", **{k: round(r[k] / 1e9, 1) for k in
            ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")},
        output_tokens=r["eval_count"])

    content = r["message"]["content"]
    logger.debug("vlm.raw_response", content=content[:500])

    if not content.strip():
        thinking = r["message"].get("thinking")
        raise RuntimeError(
            "Model returned empty content"
            + (" (it put its output in the 'thinking' field)" if thinking else "")
        )

    extraction = Extraction.model_validate_json(content)
    # Cache the result (as dict for JSON-serializable)
    vlm_cache.set(chat_model, question, page_hashes, extraction.model_dump())
    return extraction


def check_range(value, ref):
    if not ref:
        return "unknown"
    value, ref = value.replace(",", ""), ref.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", value)
    if not m:
        return "unknown"
    v = float(m.group())

    r = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)", ref)
    if r:
        lo, hi = float(r[1]), float(r[2])
        return "low" if v < lo else "high" if v > hi else "normal"
    r = re.match(r"\s*[<≤]\s*(\d+(?:\.\d+)?)", ref)
    if r:
        return "high" if v > float(r[1]) else "normal"
    r = re.match(r"\s*[>≥]\s*(\d+(?:\.\d+)?)", ref)
    if r:
        return "low" if v < float(r[1]) else "normal"
    return "unknown"


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(settings.ollama_max_retries),
    retry=retry_if_exception_type((ollama.ResponseError, llm_provider.ProviderError,
                                   ConnectionError, TimeoutError)),
    reraise=True,
)
def _explain_chat(names):
    if not _circuit_allows():
        raise RuntimeError("Circuit breaker open")
    try:
        r = llm_provider.chat(
            model=MODEL,
            messages=[{"role": "user", "content": (
                "For each medical test name, write one plain-language sentence on what it measures. "
                "No advice, no diagnosis, no mention of any values. "
                f"Return a JSON object mapping each name to its sentence. Names: {names}"
            )}],
            format="json",
            options={"num_ctx": 4096, "temperature": 0},
            keep_alive="10m",
        )
        _record_success()
        return r
    except Exception as e:
        _record_failure()
        raise


def explain_detailed(findings):
    """Build plain-language explanations with source tracking and quality scores.

    Hybrid pipeline: cache -> local SQLite dictionary -> VLM. Terms that the user
    rated 'poor' (or that cached low-quality output) are force-reexplained instead
    of trusting the dictionary/cache (feedback loop, Goal #6).

    Returns a dict with three parallel maps:
        sentences: {term: definition text}
        sources:   {term: 'local_db' | 'vlm' | 'cache'}
        qualities: {term: float 0.0-1.0 heuristic quality score}
    """
    names = list({f.name for f in findings})[:15]
    empty = {"sentences": {}, "sources": {}, "qualities": {}}
    if not names:
        return empty

    # 1. Read cache. Entries may be legacy plain strings or new meta dicts.
    cached = explain_cache.get(names) or {}
    norm = {}
    for term, val in cached.items():
        if isinstance(val, dict):
            norm[term] = {
                "definition": str(val.get("definition", "")),
                "source": val.get("source", "cache"),
                "quality": float(val.get("quality", 0.0)),
            }
        else:
            norm[term] = {
                "definition": str(val),
                "source": "cache",
                "quality": score_definition(val),
            }

    # 2. Terms the user flagged as poor, plus low-quality cache entries, get
    #    regenerated rather than trusted.
    flagged = {n for n in names if has_recent_negative_feedback(n)}
    stale = {
        n for n in names
        if norm.get(n, {}).get("quality", 0.0) < GOOD_QUALITY_THRESHOLD
    }
    regen = flagged | stale

    entries = {t: m for t, m in norm.items() if t in names and t not in regen}
    if entries:
        logger.debug("explain.cache_entries_used", terms=list(entries.keys()))

    # 3. Remaining terms: local dictionary first (skip user-flagged terms).
    missing = [n for n in names if n not in entries]
    dict_result = lookup_terms(missing)
    for term, data in dict_result["found"].items():
        if term in flagged:
            continue  # user marked this poor; force a fresh VLM definition
        entries[term] = {
            "definition": data["definition"],
            "source": "local_db",
            "quality": score_definition(data["definition"]),
        }

    # 4. Fallback to VLM for anything still uncovered (missing terms + flagged).
    remaining = [n for n in names if n not in entries]
    if remaining:
        logger.info("explain.vlm_fallback", missing=remaining, found=list(entries.keys()))
        r = _explain_chat(remaining)
        try:
            vlm_result = json.loads(r["message"]["content"])
        except Exception as e:
            logger.warning("explain.vlm_parse_failed", error=str(e))
            vlm_result = {}
        for term in remaining:
            sentence = str(vlm_result.get(term, "")).strip()
            entries[term] = {
                "definition": sentence,
                "source": "vlm",
                "quality": score_definition(sentence),
            }

    result = {
        "sentences": {t: m["definition"] for t, m in entries.items()},
        "sources": {t: m.get("source", "unknown") for t, m in entries.items()},
        "qualities": {t: round(m.get("quality", 0.0), 2) for t, m in entries.items()},
    }

    # 5. Cache full meta entries (TTL refresh each read). Legacy explain()
    #    cache entries are upgraded transparently via the normalization above.
    explain_cache.set(names, entries)
    return result


def explain(findings):
    """Backward-compatible wrapper: only the {term: definition text} map."""
    return explain_detailed(findings)["sentences"]