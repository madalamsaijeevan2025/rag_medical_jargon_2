#!/usr/bin/env python3
"""Extraction F-score harness (Goal #2 Success Metric: extraction F-score > 0.8).

Compares the CURRENT prompts vs the previous baseline prompt, term-by-term,
against a gold list you provide. Run on the machine with Ollama + models pulled.

Examples (from the project root, after the server has converted a PDF once):
  python eval_extraction_prompt.py \
      --image extracted_pages/report_page_1.png --page 1 \
      --gold "Hemoglobin,WBC,Platelets,Serum Sodium" --prompt baseline
  python eval_extraction_prompt.py \
      --image extracted_pages/report_page_1.png --page 1 \
      --gold "Hemoglobin,WBC,Platelets,Serum Sodium" --prompt current

Run both and compare the printed precision / recall / F1 to demonstrate the
Goal #2 improvement (F-score > 0.8 target).
"""
import argparse

from term_indexer import _build_extract_prompt
from vlm import extract


# The prompt used before the Goal #2 prompting upgrade (for before/after runs).
BASELINE_PROMPT = (
    "Pages are in order: {page_nums}.\n"
    "Extract ALL medical terms, lab test names, abbreviations, and clinical "
    "findings visible on these pages. For each term, include: name, value (if "
    "present), unit (if present), reference_range (if present), page number. "
    "Return JSON matching the Extraction schema. If no terms found, return "
    "empty findings array. Do not give medical advice or diagnoses."
)


def normalize(name: str) -> str:
    return " ".join(str(name).lower().split()).strip(" .,")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--image", required=True, help="Path to a page PNG (from /upload output)")
    parser.add_argument("--page", type=int, default=1, help="1-based page number of that image")
    parser.add_argument("--gold", required=True,
                        help="Comma-separated expected term names for this page")
    parser.add_argument("--prompt", choices=["current", "baseline"], default="current",
                        help="current = upgraded prompts, baseline = pre-Goal-#2 prompt")
    args = parser.parse_args()

    gold = {normalize(g) for g in args.gold.split(",") if g.strip()}
    if not gold:
        parser.error("--gold must contain at least one term")

    if args.prompt == "baseline":
        question = BASELINE_PROMPT.format(page_nums=[args.page])
    else:
        question = _build_extract_prompt([args.page])

    print(f"prompt={args.prompt}  page={args.page}  image={args.image}")
    result = extract(question, [{"page": args.page, "image_path": args.image}])

    got = {normalize(f.name) for f in result.findings if normalize(f.name)}
    matched = got & gold
    missed = gold - got
    extra = got - gold

    precision = len(matched) / len(got) if got else 0.0
    recall = len(matched) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nextracted={len(got)}  gold={len(gold)}  matched={len(matched)}")
    print(f"  matched : {sorted(matched)}")
    print(f"  missed  : {sorted(missed)}")
    print(f"  extra   : {sorted(extra)}")
    print(f"\nprecision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
    print("goal: F1 > 0.80 -> " + ("ACHIEVED" if f1 > 0.80 else "not yet achieved"))


if __name__ == "__main__":
    main()
