# Medical-jargon-explainer: Project Roadmap & Future Goals

## Vision
A document intelligence service that extracts, indexes, and explains medical terminology from PDFs using VLMs, local dictionaries, and semantic search.

---

## 🎯 Priority Goals (Q1-Q2)

### 1. Expand Local Medical Dictionary
- **Goal**: Add 200+ core medical terms to `seed_medical_db.py`
- **Target**: Increase dictionary hit rate from ~30% to ~70% for common terms
- **Actions**:
  - Add terms across categories: lab markers, radiology, conditions, procedures, anatomy
  - Include common abbreviations (e.g., "CBC", "DNA", "MRI")
  - Add clinical significance fields with reference ranges
- **Success Metric**: `lookup_terms()` returns "found" for 70% of terms extracted by VLM

### 2. Improve VLM Extraction Prompting
- **Goal**: Refine prompts in `vlm.py` and `term_indexer.py`
- **Target**: Reduce missed terms and false positives by 40%
- **Actions**:
  - Add section headers detection (e.g., "Laboratory Results", "Impressions")
  - Improve handling of tables and formatted text
  - Add support for multi-column layouts common in medical reports
- **Success Metric**: Extraction F-score > 0.8 on test medical PDFs

### 3. Add Hybrid Search Capability
- **Goal**: Combine vector similarity with keyword search in `retriever.py`
- **Target**: Improve recall for relevant pages
- **Actions**:
  - Implement TF-IDF or BM25 alongside ChromaDB cosine similarity
  - Add reranking function that combines both scores
  - Support search by term category (lab marker vs condition vs radiology)
- **Success Metric**: Search results include 20% more relevant pages

### 4. Web Frontend (Streamlit)
- **Goal**: Build minimal Streamlit interface using existing `streamlit` dependency
- **Target**: Enable non-technical users to upload and query documents
- **Actions**:
  - Document upload widget
  - Term list per document
  - Simple question-answer interface
  - Term explanation display
- **Success Metric**: Upload → query workflow completes in < 30 seconds

---

## 📈 Medium-Term Goals (Q3-Q4)

### 5. Multilingual Support
- **Goal**: Add term translation and cross-language search
- **Target**: Support English + Spanish medical terms initially
- **Actions**:
  - Integrate Ollama for term translation
  - Add language field to medical_terms table
  - Update `_is_valid_term()` to accept multilingual categories
- **Success Metric**: Terms in Spanish/French can be looked up and explained

### 6. Explanation Quality Enhancement
- **Goal**: Improve `explain()` function in `vlm.py`
- **Target**: More accurate plain-language definitions
- **Actions**:
  - Add source tracking (local DB vs VLM vs external API)
  - Implement explanation quality scoring
  - Add user feedback loop to re-explain poor definitions
- **Success Metric**: User satisfaction score > 4/5 for explanations

### 7. Document Export & Reporting
- **Goal**: Add export functionality from `main.py`
- **Target**: Generate PDF/CSV reports of findings
- **Actions**:
  - Export findings as CSV/JSON
  - Generate simple PDF report with term definitions
  - Include page references and original values
- **Success Metric**: Export generates valid file in < 5 seconds

### 8. User Authentication & Authorization
- **Goal**: Add basic user system
- **Target**: Document ownership and privacy
- **Actions**:
  - Add API key middleware to FastAPI
  - User-owned document buckets
  - Rate limiting per API key
- **Success Metric**: Auth middleware blocks unauthenticated requests

---

## 🏗️ Technical Debt & Polish

### 9. Testing Suite
- **Goal**: Add integration and unit tests
- **Target**: >80% code coverage for critical paths
- **Actions**:
  - Expand `smoke_test.py` with realistic medical PDFs
  - Add unit tests for `check_range()`, `_parse_terms_from_text()`
  - Add integration test: upload → index → search → query → explain
- **Success Metric**: `pytest` passes with no failures

### 10. Performance Optimizations
- **Goal**: Reduce VLM API costs and latency
- **Target**: 50% reduction in per-query costs
- **Actions**:
  - Implement request batching for similar questions
  - Add cache warming on startup
  - Set maximum context window limits
  - Add concurrent request limiting
- **Success Metric**: Average query time < 8 seconds (excluding VLM latency)

### 11. Monitoring & Observability
- **Goal**: Add structured metrics and alerts
- **Target**: Production-ready observability
- **Actions**:
  - Add Prometheus metrics endpoint
  - Track VLM token usage and costs
  - Monitor cache hit rates
  - Alert on Ollama/ChromaDB connectivity loss
- **Success Metric**: Grafana dashboard shows system health

### 12. API Versioning
- **Goal**: Add versioned endpoints
- **Target**: Backward compatibility
- **Actions**:
  - Prefix endpoints with `/api/v1/`
  - Deprecation warnings for old endpoints
  - Changelog generation
- **Success Metric**: v1 endpoints work; v0 continues with deprecation notice

---

## 📊 Success Metrics Dashboard

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Dictionary hit rate | ~30% | 70% | Q1 |
| Extraction F-score | ~0.65 | 0.80 | Q1 |
| Search recall | baseline | +20% | Q2 |
| Avg query time | N/A | < 8s | Q3 |
| Code coverage | 0% | 80% | Q4 |
| VLM cost per query | N/A | 50% reduction | Q3 |

---

## 🚀 Next Immediate Action

**Start with**: Add 50+ medical terms to `seed_medical_db.py` and run `python seed_medical_db.py` to verify database initialization.

This will immediately improve the `explain()` function's dictionary fallback rate and reduce VLM calls for common terms.

---

*Roadmap last updated: October 14, 2026*