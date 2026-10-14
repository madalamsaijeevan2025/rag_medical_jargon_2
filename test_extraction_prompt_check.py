#!/usr/bin/env python3
"""Offline checks for improved VLM extraction prompting (Goal #2).

No Ollama needed: chat calls are faked (the flow still exercises the real
prompt assembly and JSON-schema parsing). Verifies that the guidance reaches
every extraction prompt and that the term parser drops document noise.
Run from the project root: python test_extraction_prompt_check.py
"""
import json

import vlm
import term_indexer as ti


class FakeFinding:
    def __init__(self, name):
        self.name = name


NOISY_THINKING_TEXT = "\n".join([
    "Wait, let me check the report format first...",
    "Patient Name: John Doe",
    "DATE: 12/03/2025",
    "Accession: 99213-C",
    "Hemoglobin | Lab Marker",
    "- Creatinine | Lab Marker",
    "eGFR → Lab Marker",
    "Troponin-I | Lab Marker",
    "122.4",
])


def fake_chat(**kwargs):
    """Capture kwargs and reply with a valid Extraction JSON for one finding."""
    fake_chat.kwargs = kwargs
    content = json.dumps({
        "answer": "The hemoglobin result is 9.8 g/dL, which is below the reference range.",
        "findings": [{
            "name": "Hemoglobin", "value": "9.8", "unit": "g/dL",
            "reference_range": "12-16", "flag": "LOW", "page": 1, "bbox": [],
        }],
    })
    return {
        "message": {"content": content},
        "total_duration": 0, "load_duration": 0,
        "prompt_eval_duration": 0, "eval_duration": 0,
        "eval_count": 0,
    }


def _make_tiny_png(path: str):
    """Real 1x1 image: extract() hashes the actual file for its cache key."""
    from PIL import Image
    Image.new("RGB", (16, 16), "white").save(path)


def main():
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"{name}: {status}")
        if not cond:
            failures.append(name)

    # --- 1. Guidance constants exist and are non-trivial ---
    check("SECTION_HINTS populated", len(vlm.SECTION_HINTS) >= 10)
    check("LAB_TERM_HINTS populated", len(vlm.LAB_TERM_HINTS) >= 20)
    check("PAGE_LAYOUT_GUIDANCE non-empty", len(vlm.PAGE_LAYOUT_GUIDANCE) > 200)

    # --- 2. Guidance reaches the term_indexer extraction prompt ---
    prompt = ti._build_extract_prompt([1, 2])
    check("indexer prompt: page order kept", prompt.startswith("Pages are in order: [1, 2]."))
    check("indexer prompt: layout guidance included", vlm.PAGE_LAYOUT_GUIDANCE in prompt)
    check("indexer prompt: high-priority sections listed",
          "HIGH-PRIORITY sections:" in prompt and "Laboratory Results" in prompt)
    check("indexer prompt: anchor terms listed",
          "ANCHOR TERMS" in prompt and "Hemoglobin" in prompt)
    check("indexer prompt: noise-ignore rule present", "IGNORE as noise" in prompt)

    # --- 3. Guidance reaches the /query extraction prompt (extract()) ---
    orig_chat, orig_shrink = vlm._chat_with_retry, vlm._shrink
    vlm._shrink = lambda path, max_side=1400: b"fake-png-bytes"
    img_path = "test_min.png"
    _make_tiny_png(img_path)
    # VLM answers are cached on disk by (model, question, page-file hash). The
    # helper PNG is byte-identical between runs, so drop the test's cache entry
    # to guarantee extract() actually exercises prompt assembly + fake chat.
    vlm.vlm_cache._backend.delete(
        vlm.vlm_cache.key(vlm.MODEL, "what is low?", [vlm._page_hash(img_path)])
    )
    try:
        vlm._chat_with_retry = fake_chat
        result = vlm.extract("what is low?", [{"page": 1, "image_path": img_path}])
        check("extract(): schema reply parsed",
              result.findings[0].name == "Hemoglobin" and result.answer)
        content = fake_chat.kwargs["messages"][0]["content"]
        check("query prompt: layout guidance included", vlm.PAGE_LAYOUT_GUIDANCE in content)
        check("query prompt: question preserved", content.startswith("Question: what is low?"))
        check("query prompt: reference-range copy rule kept",
              "Copy the reference range" in content)
    finally:
        vlm._chat_with_retry, vlm._shrink = orig_chat, orig_shrink

    # --- 4. Terms-from-image prompt also upgraded ---
    orig_chat2 = vlm._chat_with_retry
    vlm._chat_with_retry = fake_chat
    try:
        vlm.extract_terms_from_image(img_path)
        content = fake_chat.kwargs["messages"][0]["content"]
        check("image-terms prompt: layout guidance included",
              vlm.PAGE_LAYOUT_GUIDANCE in content)
        check("image-terms prompt: sections prioritized",
              "section headers such as" in content)
    finally:
        vlm._chat_with_retry = orig_chat2

    # --- 5. Parser hardening on noisy VLM 'thinking' output ---
    terms = {t["name"] for t in vlm._parse_terms_from_text(NOISY_THINKING_TEXT)}
    check("noise (patient/date/accession) excluded",
          all(n not in terms for n in ("John Doe", "12/03/2025", "99213-C")))
    check("clinical terms kept", terms == {"Hemoglobin", "Creatinine", "eGFR", "Troponin-I"})

    # --- 6. _is_valid_term guards ---
    check("bare value rejected", vlm._is_valid_term("12.5", "Lab Marker") is False)
    check("bare percent rejected", vlm._is_valid_term("97.2%", "Lab Marker") is False)
    check("patient-id line rejected", vlm._is_valid_term("Patient Name", "Other") is False)
    check("real term accepted", vlm._is_valid_term("Hemoglobin", "Lab Marker") is True)
    check("new category accepted", vlm._is_valid_term("Hemoglobin", "hematology") is True)
    check("existing category still accepted", vlm._is_valid_term("eGFR", "Lab Marker") is True)

    import os
    try:
        os.remove(img_path)  # 1-pixel helper image; recreated on next run
    except OSError:
        pass

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
