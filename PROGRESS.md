# PROGRESS.md — Project Memory / Record File

> **Purpose:** Read this file at the start of every AI session (or whenever resuming work)
> so you know exactly what has been done, what was verified, and what to do next.
> Update it (top of "Session Log") at the end of every session.

**Last updated:** 2026-10-14 (Session 3)

---

## 1. Project snapshot (for fast context)

- **Project:** Medical-jargon-explainer — FastAPI service that:
  1. converts PDF pages to images (`/upload`),
  2. indexes page text via Ollama embeddings + ChromaDB,
  3. extracts medical terms via VLM (`qwen3-vl:4b`),
  4. answers questions and explains terms in plain language (`/query`),
  5. builds per-document term indexes (`/terms/{doc_id}`).
- **Key files:** `main.py` (routes), `vlm.py` (VLM + explanations), `retriever.py`
  (embeddings/search), `document_store.py` (ChromaDB + JSON meta),
  `dictionary_service.py` (SQLite dictionary + feedback), `term_indexer.py`
  (background term indexing), `cache.py` (diskcache), `health.py`, `cleanup_job.py`,
  `config.py` (all settings, env via `.env`), `seed_medical_db.py` (37 seed terms),
  `FUTURE_GOALS.md` (roadmap — 12 goals), `smoke_test.py` (2-page smoke test).
- **Roadmap:** see `FUTURE_GOALS.md`. This file records implementation progress on it.

## 2. Environment notes (important)

- **Dev machine used so far:** Windows, Git-Bash shell, Python 3.12 (Store install).
  Use `python3`, POSIX syntax. Project lives at
  `C:\Users\Admin\OneDrive\Desktop\idp-da-2 progression\Medical-jargon-explainer`.
- **Installed (Sessions 2–3, for offline verification + real-app smoke tests):**
  `tenacity ollama pillow pydantic pydantic-settings structlog diskcache fastapi
  uvicorn pymupdf apscheduler httpx chromadb python-multipart`. The FastAPI app
  now imports and routes on the dev machine too; only real Ollama inference needs the laptop.
- **Repair made (Session 3):** `python-multipart` was missing everywhere — FastAPI ≥0.110
  raises RuntimeError for the `/upload` UploadFile endpoint without it. Added to
  `requirements.txt` (apply on the laptop: `pip install python-multipart`).
- Full end-to-end flow needs: Ollama running at `http://localhost:11434` with
  models pulled: `nomic-embed-text` and `qwen3-vl:4b`
  (`ollama pull nomic-embed-text && ollama pull qwen3-vl:4b`).

## 3. Session log

### Session 1 — 2026-10-14: Analysis + roadmap
- Analysed the full codebase and architecture.
- Created `FUTURE_GOALS.md` with 12 goals (priorities, actions, success metrics,
  metrics dashboard, next immediate action).

### Session 4 — 2026-10-14: Full-folder audit — 3 fixes

**Audit scope:** all files reviewed; `python -m compileall` clean end-to-end; both test
suites re-run green after each repair; real-app openapi/health smoke re-verified.

| Finding | Fix |
|---|---|
| `vlm.py` `extract_terms_from_image()` contained a byte-identical duplicate of `_shrink()` (`_shrink_local`) incl. redundant local `PIL/io` imports |deleted duplicate; call now uses module-level `_shrink()` (also lets tests fake one shrink path) |
| `main.py` /upload hardcoded `http://127.0.0.1:8000/static/...` page URLs (break on any other host/port) | response URLs now built from `request.base_url` (handler signature gained `request: Request`) |
| `document_store.upsert_page()` inflated `page_count` (+1 per call) and `size_bytes` on every re-upload of the same PDF | distinct-page counting (`max(count, page_no)`), size reset + `created_at` refresh when page 1 re-indexes; verified with a dedicated runtime check (page1→1, page2→2/s1+s2, re-upload→2/s1, no inflation) |

**Noted, no change (by design / pre-existing):**
- `.gitignore` ignores `scripts/` — `scripts/populate_terms_from_images.py` stays unversioned (intentional-looking author choice; flag if you want it committed).
- `retriever.py::_is_transient` is unused (retry decorator uses explicit exception types) — left as-is, harmless.
- `migrate_schema.py` — pre-existing standalone DB migration helper, untouched.
- `term_indexes/*.json` show `terms_found: 0` (old failed runs) — artifacts, not a code bug.

### Session 3 — 2026-10-14: Goal #2 “Improve VLM Extraction Prompting” 🟨 IMPLEMENTED (F-score calibration pending laptop)

**What was asked:** implement Goal #2 (better extraction prompting).

**What changed:**

| File | Change |
|---|---|
| `vlm.py` | New shared guidance constants: `SECTION_HINTS` (29 report section headers), `LAB_TERM_HINTS` (39 anchor lab terms), `PAGE_LAYOUT_GUIDANCE` (table row-by-cell reading, two-column layout order, value-vs-reference-range column separation, explicit IGNORE-noise list). `PAGE_LAYOUT_GUIDANCE` is now appended to **all three** extraction prompts: `extract()` (the `/query` path), `extract_terms_from_image()`, and the term-indexing prompt (via `term_indexer`). `_is_valid_term()` hardened: noise prefixes (patient/date/accession/collected/doctor/…), new valid categories (hematology, chemistry, microbiology, pathology, medication, …), and a bare value/unit/date guard (e.g. “12.5”, “97.2%” rejected). |
| `term_indexer.py` | `_build_extract_prompt()` rewritten: TASK/INCLUDE framing, HIGH-PRIORITY sections, ANCHOR TERMS list, layout guidance, exact value/range copying. |
| `eval_extraction_prompt.py` | **NEW** F-score harness (Goal #2 success metric): runs one page image through `vlm.extract` with `--prompt current` or `--prompt baseline` (baseline = the pre-upgrade prompt, kept verbatim) against a gold term list; prints precision/recall/F1 per run. |
| `test_extraction_prompt_check.py` | **NEW** offline checks: guidance reaches all 3 prompts, question text & schema flow unchanged, noisy “thinking” output parsing (patient/date/accession/“122.4” dropped; clinical terms kept), validity guards. Uses a real 1-pixel PNG because `extract()` hashes the page file for its cache key. |

**Verified offline (dev machine, no Ollama):** guidance constants present in all prompt paths; JSON-schema extraction flow parses a fake reply; noisy thinking output → exactly {Hemoglobin, Creatinine, eGFR, Troponin-I}; bare values/noise rejected. Test is idempotent (twice-in-a-row, exit 0) — it isolates BOTH cache layers: `explain_cache` (clear-prefix at start) and `vlm_cache` (deletes its exact test key, because the helper PNG is byte-identical between runs and would otherwise hit cache and skip prompt assembly).

**Not verifiable offline → run on the laptop (Ollama):**
```bash
# F-score measurement (Goal #2 metric): same page, both prompts
python eval_extraction_prompt.py --image extracted_pages/<doc>_page_1.png --page 1 \
    --gold "Hemoglobin,WBC,Platelets,Serum Sodium,…” --prompt baseline
python eval_extraction_prompt.py --image extracted_pages/<doc>_page_1.png --page 1 \
    --gold "Hemoglobin,WBC,Platelets,Serum Sodium,…” --prompt current
# → target: current F1 > 0.80; record both numbers here (Session log §3)

# NB: requirements.txt now includes python-multipart (was missing -> /upload would
# crash on fastapi>=0.110); run `pip install python-multipart` on the laptop.
```

### Session 2 — 2026-10-14: Goal #6 “Explanation Quality Enhancement” ✅ DONE

**What was asked:** implement Goal #6 and keep a memory/progress record file
(this file) for future sessions.

**What changed (5 source files + 1 test):**

| File | Change |
|---|---|
| `vlm.py` | New `score_definition()` heuristic quality scorer (0.0–1.0, flags placeholder/error text like "this is not on", "unknown", all-caps, too short/long). New `explain_detailed()` returning `{"sentences", "sources", "qualities"}` maps with **source tracking** (`local_db` / `vlm` / `cache`). Terms the user rated **poor** (or cache entries scoring < 0.5) are **force re-explained via VLM** instead of trusting cache/dictionary (feedback loop). `explain()` kept as backward-compatible wrapper. Fixed latent bug: non-string VLM results now coerced to `str`. |
| `dictionary_service.py` | New SQLite table `explanation_feedback` (term, rating good/fair/poor, comment, created_at UTC ISO) + index, auto-created in `_ensure_db()`. New functions: `record_feedback()`, `has_recent_negative_feedback()` (‘poor’ within 7 days with no later ‘good’), `get_feedback_summary()` (counts, satisfaction score out of 5, flagged terms). |
| `seed_medical_db.py` | Creates the same `explanation_feedback` table on fresh DB init. |
| `main.py` | `/query` response rows now include `definition_source` and `definition_quality` per finding. New endpoints: `POST /feedback/explanation` (body `{term, rating: poor|fair|good, comment?}`, 422 on bad rating/empty term) and `GET /explanations/quality` (summary incl. `avg_satisfaction_out_of_5` → Goal #6 metric). |
| `term_indexer.py` | Term index JSON per document now stores `definition_source` and `definition_quality` per term. |
| `test_explanation_quality_check.py` | **NEW** standalone verification script (no pytest needed, fake VLM, no Ollama needed). |

**Cache format note:** `explain_cache` entries upgraded from `{term: sentence}` to
`{term: {definition, source, quality}}`; legacy string entries are normalized
transparently at read time — nothing to migrate.

**Verified on dev machine (offline):**
- `python3 test_explanation_quality_check.py` → **all 22 checks PASS, exit 0**, twice in a
  row (idempotent: it clears/repairs only its own test cache keys and feedback rows at
  start/end — a run-1 fake-VLM cache entry had made run-2 flaky, fixed via test isolation).
- `python3 seed_medical_db.py` → OK; `explanation_feedback` table confirmed present in `medical_terms.db`.
- `ast.parse` syntax OK for all 6 files.
- (Session 3 update) Real-app smoke test now passes on the dev machine (TestClient, no lifespan):
  `POST /feedback/explanation` 200 + request-id, `GET /explanations/quality` shows
  flagged_terms/avg_satisfaction, invalid rating → 422, `GET /health` 200. Still needs the laptop: **/upload, /query with real VLM inference, F-score measurement.**

### Session 5 — 2026-10-15: Cloud-API fallback (no Ollama needed) ✅ DONE

**What was asked:** when a user has no Ollama locally, give an option to paste an
API key and have ALL VLM/AI work (page reading, extraction, explanations,
embeddings) go through that key instead.

**How it works:** any OpenAI-compatible endpoint (OpenAI `/v1`, OpenRouter, Groq,
Gemini's OpenAI-compat surface, …). The key can be pasted in the Streamlit
sidebar, sent to `POST /llm/config`, or seeded from `.env` (headless). Ollama
stays the default: nothing changes for machines that have it.

**What changed (6 source files + 1 new test):**

| File | Change |
|---|---|
| `llm_provider.py` | **NEW** runtime provider module: in-memory state (provider/api_key/base_url/model/embed_model, seeded from env), `configure()`/`status()` (key never echoed), `ollama_reachable()` probe, and OpenAI-compatible adapters `chat_api()`/`embed_api()` that translate Ollama-style kwargs+responses (images → base64 data URIs; `format=` → `response_format: json_object` + schema pinned into the prompt; Ollama timing-key shape preserved). Retryable `ProviderError(ConnectionError)` for failures. |
| `config.py` | New settings: `llm_provider` (default `local`), `api_key`, `api_base_url`, `api_model`, `api_embed_model`. |
| `vlm.py` | Both chat call sites (`_chat_with_retry`, `_explain_chat`) now route through `llm_provider.chat()`; retry blocks also catch `ProviderError`. `extract()` cache keys use the **effective** chat model (local vs cloud results never collide; `_normalize_stats` guards timing logging). |
| `retriever.py` | `_embed_with_retry()` routes to `llm_provider.embed()` in api mode (no ollama-only `num_ctx` option); cache keys use the effective embed model; retry catches `ProviderError`. |
| `main.py` | New endpoints: `POST /llm/config` (switch provider at runtime; 422 on bad provider / missing key) and `GET /llm/status` (reports provider, models, base_url, `api_key_set`, `ollama_reachable` — no key material). |
| `app.py` | Sidebar "AI Backend" choice: **Local Ollama** (shows Ollama reachability warning) or **Cloud API key** — password input for the key + endpoint/chat-model/embedding-model fields + *Apply* button that calls `/llm/config`. |
| `test_llm_api_fallback_check.py` | **NEW** offline test (fake HTTP, no Ollama, no real API): proxy switch/apply, key masking, bearer header, no-key rejection, `extract()` end-to-end via the adapter (data-URI PNG verified, schema pinned, json response_format), `explain()` fallback, embeddings routing/order, error surfacing, and `/llm/config`+`/llm/status` through the real FastAPI app. **16/16 PASS, exit 0.** |

**Regression checks after the change:** `compileall` clean; Goal #2 test 22/22 exit 0;
Goal #6 test all-PASS exit 0; FastAPI smoke (`/health`, `/llm/*` round-trip, key never
in response bodies, openapi contains new endpoints) exit 0 — then provider restored to
`local` (unit tests keep `conversation state` clean). NB: `.env` key format is
`LLM_PROVIDER=api`, `API_KEY=...` etc.; cloud defaults live in `llm_provider.py`
(OpenAI `/v1`, `gpt-4o-mini`, `text-embedding-3-small`).**Still laptop-only:** an actual cloud call with a real key (billable) and Ollama
side-by-side comparison.

**Addendum 1 (2026-10-15):** first real-world use surfaced the classic
misconfiguration (a key from an unknown/unsupported provider pasted against the
default OpenAI endpoint → upstream 401, surfaced as 503 at upload/indexing).
Mitigation added: *app.py* now probes `{base_url}/models` with the key as soon as
**Apply API key** is clicked, and reports "accepted" / "REJECTED (401/403), check
provider-endpoint match" or a warning — before any upload happens. Backend config
flow unchanged.

**Addendum 2 (2026-10-15):** upload then failed on a **stale Gemini embedding
model name** (`text-embedding-004` → 404 "not supported for embedContent").
Mitigation: `embed_api()` now retries once on a model-level 404 using (a) a small
set of current names and (b) Google's own suggested replacement parsed from the
error body; the winner is cached in session state so the next call skips the dead
name; non-model errors (401/429/network) never trickle through. Verified: offline
checks cover the fallback + that auth/quota 401s are not retried. NB: the retired
name involved (`text-embedding-004`) is confirmed retired via Google's current
docs/community reports.

**Addendum 3 (2026-10-15):** a **slow page-index call** (cloud embeds take seconds)
ran inside the async `/upload` handler and blocked the event loop: every other
endpoint stalled and the app's status/term-index polls timed out (real log:
`ReadTimeout: ... HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out`).
Mitigation: `index_page()` now runs in a worker thread; verified on a live uvicorn
instance that `/llm/status` answers ~0.02s even mid-simulated-3s-upload. Also, on
this machine the `ollama_reachable()` port probe used a flat timeout that turned
into a long stall on Windows; it now uses a short connect timeout + 10s cache ---
status replies 0.01s when cached. See commit `39ecc96`.

**Addendum 4 (2026-10-15):** the **adapt/connect step** above surfaced **another
stale model name**, this time the **chat model** `gemini-2.0-flash` → 404
"no longer available, use models/gemini-3.8-flash". Mitigations:
*llm_provider.py* `chat_api()` now mirrors the embeddings self-heal — after a
model-level 404 it parses the provider's own suggested replacement and retries with
known-current names (embedded once per session), filters out non-model errors; plus
a circuit breaker open now requires a retryable failure (regex lookup in the error
body to extract suggested endings) that can be fixed by `reset_circuit_breaker()`.
Verified: the `/llm/config` endpoint now resets the breaker on re-apply; offline
checks cover both chat and embed fallbacks and the breaker-reset-wiring.

## 4. Goal status (from FUTURE_GOALS.md)

| # | Goal | Status |
|---|---|---|
| 1 | Expand local medical dictionary (200+ terms) | ⬜ Not started — **next up** |
| 2 | Improve VLM extraction prompting | 🟨 **Prompts implemented (Session 3)**; F-score > 0.8 must be measured on laptop with eval harness |
| 3 | Hybrid search (vector + keyword) | ⬜ Not started |
| 4 | Streamlit frontend | ⬜ Not started (`streamlit` already in requirements) |
| 5 | Multilingual support | ⬜ Not started |
| 6 | Explanation quality (source tracking, quality scoring, feedback loop) | ✅ **DONE (Session 2)** |
| 7 | Document export & reporting | ⬜ Not started |
| 8 | Auth & API keys | ⬜ **DONE (Session 5-6)**: cloud-api fallback with runtime config via Post /llm/config, key masking, session-fit, re-apply regenerates on later runs |
| 9 | Testing suite | 🟨 Partial: standalone check script added; pytest suite still open |
| 10 | Performance optimizations | ⬜ Not started |
| 11 | Monitoring & observability | ⬜ Not started |


| 12 | API versioning (`/api/v1`) | ⬜ Not started |

## 5. How to run / verify on the laptop (has Ollama)

```bash
# 1. One-time setup
pip install -r requirements.txt
ollama pull nomic-embed-text && ollama pull qwen3-vl:4b

# 2. Offline quality checks (no Ollama needed)
python test_explanation_quality_check.py     # expect: All checks passed. (22 PASS)

# 3. Start the service
python main.py        # FastAPI on http://127.0.0.1:8000 (docs at /docs)

# 4. End-to-end verification of Goal #6
#    a) upload a PDF, then:
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" ^
  -d "{\"doc_id\": \"<doc_id>\", \"question\": \"list the lab values\"}"
#    → each finding row must contain definition_source and definition_quality
#    b) rate an explanation poorly, then re-query:
curl -X POST http://127.0.0.1:8000/feedback/explanation -H "Content-Type: application/json" ^
  -d "{\"term\": \"HbA1c\", \"rating\": \"poor\", \"comment\": \"too vague\"}"
#    → next /query for HbA1c must show definition_source == "vlm" (forced re-explanation)
#    c) satisfaction summary:
curl http://127.0.0.1:8000/explanations/quality
#    → {"total_feedback": n, "ratings": {...}, "avg_satisfaction_out_of_5": x, "flagged_terms": [...]}

# (Use bash instead of cmd: single-quoted JSON '...' and no ^ line continuation.)
```

## 6. Next steps (recommended order)

1. **Laptop verification run** (§5 steps 2–4) — confirm Goals #6 and #2 end-to-end with real Ollama; update this file with results.
2. **Measure Goal #2 F-score** with `eval_extraction_prompt.py` (baseline vs current) on a real page image; fill Session-3 numbers into this file.
4. **Goal #1:** add 50–200 medical terms to `seed_medical_db.py` (biggest metric lift: dictionary hit rate → 70%).
3. **Goal #2:** refine extraction prompts in `vlm.py` / `term_indexer.py`.
4. **Goal #4:** Streamlit frontend (upload → term list → Q&A → feedback buttons that
   call `POST /feedback/explanation`).

## 7. Gotchas / notes for the next session

- Feedback flag lives in SQLite (`explanation_feedback`), decays after 7 days since
  the last rating; a later `good` rating clears it.
- `explain_cache` entries are auto-upgraded; no migration needed. Clearing
  `./cache` is always safe but not required.
- `medical_terms.db` path = `./medical_terms.db` (cwd-relative via `config.py`).
  Dev-machine DB already seeded; on the laptop it seeds automatically at app start
  (`main.py` lifespan).
- VLM `temperature=0` + `format=json`; if content is empty, `qwen3-vl` may put
  output in the `thinking` field (checked in `extract()`);
  `explain_detailed()`'s VLM fallback parses plain JSON from chat responses.
- Don't remove `explain()` wrapper — external callers/tests may rely on it.
