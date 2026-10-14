#!/usr/bin/env python3
"""Standalone checks for Explanation Quality Enhancement (Goal #6).

Run from the project root: python test_explanation_quality_check.py
Uses a fake VLM (no Ollama needed) and cleans up its test rows afterwards.
"""
import json

import vlm
import dictionary_service as ds


class FakeFinding:
    def __init__(self, name):
        self.name = name


def main():
    failures = []

    # Test isolation: clear cached explanations so the 'local_db source' path is
    # exercised from a clean slate (mirrors dictionary_service's own clear_prefix
    # invalidation pattern). Safe: cache entries are recomputed on demand.
    vlm.explain_cache._backend.clear_prefix("explain:")

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"{name}: {status}")
        if not cond:
            failures.append(name)

    # --- 1. score_definition heuristics ---
    check("empty definition scores 0", vlm.score_definition("") == 0.0)
    check("placeholder text scores 0",
          vlm.score_definition("This is not on the pages I read.") == 0.0)
    check("error text scores 0",
          vlm.score_definition("Unknown term: unable to determine meaning.") == 0.0)
    check("short gibberish scores low", vlm.score_definition("abc") <= 0.4)
    good = "Measures average blood glucose over the preceding two to three months."
    check("good definition scores high", vlm.score_definition(good) >= 0.6)

    # --- 2. Feedback loop basics ---
    check("invalid rating rejected", ds.record_feedback("zzT1", "terrible") is False)
    ds.record_feedback("zzT1", "poor", "gibberish definition")
    check("poor => flagged", ds.has_recent_negative_feedback("zzT1") is True)
    ds.record_feedback("zzT1", "good", "fixed")
    check("later good clears flag", ds.has_recent_negative_feedback("zzT1") is False)
    check("unknown term not flagged", ds.has_recent_negative_feedback("zzNeverSeen") is False)

    # --- 3. explain_detailed: local dictionary source path ---
    res = vlm.explain_detailed([FakeFinding("eGFR")])
    check("eGFR explained", bool(res["sentences"].get("eGFR")))
    check("eGFR source is local_db", res["sources"].get("eGFR") == "local_db")
    check("eGFR quality high", res["qualities"].get("eGFR", 0) >= 0.6)
    # second call: served from cache meta entries, still valid
    res_b = vlm.explain_detailed([FakeFinding("eGFR")])
    check("cached entry keeps source", res_b["sources"].get("eGFR") == "local_db")

    # --- 4. Feedback forces VLM re-explanation ---
    def fake_explain(names):
        content = json.dumps({
            n: "Fake plain-language sentence used only for offline verification."
            for n in names
        })
        return {"message": {"content": content}}

    original_explain_chat = vlm._explain_chat
    vlm._explain_chat = fake_explain
    try:
        ds.record_feedback("eGFR", "poor", "nightly check")
        res2 = vlm.explain_detailed([FakeFinding("eGFR")])
        check("flagged term forced to VLM", res2["sources"].get("eGFR") == "vlm")
        check("VLM re-definition scored", res2["qualities"].get("eGFR", 0) >= 0.6)
        # backwards-compatible wrapper
        check("explain() wrapper works", bool(vlm.explain([FakeFinding("eGFR")]).get("eGFR")))
    finally:
        vlm._explain_chat = original_explain_chat

    # --- 5. Legacy cache entries (plain strings) are normalized ---
    vlm.explain_cache.set(["zzLegacyGood"], {
        "zzLegacyGood": "Blood urea nitrogen test measuring waste in the blood."
    })
    res3 = vlm.explain_detailed([FakeFinding("zzLegacyGood")])
    check("legacy string entry used", bool(res3["sentences"].get("zzLegacyGood")))
    check("legacy quality scored", res3["qualities"].get("zzLegacyGood", 0) >= 0.6)
    check("legacy source marked cache", res3["sources"].get("zzLegacyGood") == "cache")

    # --- 6. Feedback summary shape ---
    summary = ds.get_feedback_summary()
    check("summary has counts", isinstance(summary.get("ratings"), dict))
    check("summary has satisfaction score",
          summary.get("avg_satisfaction_out_of_5") is None or
          isinstance(summary.get("avg_satisfaction_out_of_5"), (float, int)))
    check("summary lists flagged terms", isinstance(summary.get("flagged_terms"), list))

    # --- cleanup test rows and test cache keys ---
    conn = ds._get_conn()
    conn.execute("DELETE FROM explanation_feedback WHERE term LIKE 'zz%'")
    conn.execute("DELETE FROM explanation_feedback WHERE term = 'eGFR'")
    conn.commit()
    for names in (["eGFR"], ["zzLegacyGood"]):
        vlm.explain_cache._backend.delete(vlm.explain_cache.key(names))

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
