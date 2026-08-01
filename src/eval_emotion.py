#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Qualitative evaluation harness for the performance script.

The pass/fail regression guardrails live in tests/test_emotion_distribution.py;
this is the human-eyeball counterpart: it prints the emotion distribution,
attribution-reduction breakdown with real examples, and character-aware delivery
deltas for one or more corpus books, so a person can sanity-check the outputs
that automated thresholds can't fully judge.

    python -m src.eval_emotion "The Red-Headed League" "Frankenstein"

It was written to catch exactly the failure it now guards against: a detective
novel reading as 80% "Cheery" because a character prior snowballed. Keep it
runnable -- re-run it whenever the detector, the lexicon, or the expression
layer changes.
"""

import argparse
import collections
import json
import os

from src.attribution_reduction import build_performance_script
from src.book_structure_adapter import load_line_payloads, load_structure, structure_to_manifest


def _build(book: str) -> str:
    structure = load_structure(book)
    manifest = structure_to_manifest(structure, line_payloads=load_line_payloads(book))
    tier3 = os.path.join("data", "corpus", "pipeline", book, "tier3")
    os.makedirs(tier3, exist_ok=True)
    manifest_path = os.path.join(tier3, "canonical_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f)
    return build_performance_script(manifest_path)


def evaluate(book: str) -> None:
    print("=" * 78)
    print(f"BOOK: {book}")
    print("=" * 78)
    data = json.load(open(_build(book), encoding="utf-8"))
    reductions = data["reductions"]
    lines = [l for s in data["scenes"] for l in s["lines"]]
    total = max(1, len(lines))

    by_action = collections.Counter(r["action"] for r in reductions)
    print(f"\nLines: {len(lines)}   Attribution reductions: {len(reductions)}  {dict(by_action)}")

    def sample(pred, n=4):
        return [r for r in reductions if pred(r)][:n]

    print("\n  REMOVED (redundant 'X said'):")
    for r in sample(lambda r: r["action"] == "remove"):
        print(f"     {r['speaker']!r} <- \"{r['attribution_text'][:64]}\"")
    print("  METADATA (expressive verb -> delivery):")
    for r in sample(lambda r: r["action"] == "metadata"):
        print(f"     delivery={r['delivery']!r} <- \"{r['attribution_text'][:56]}\"")

    emo = collections.Counter(l.get("emotion", "?") for l in lines)
    print("\n  Emotion distribution:")
    for label, count in emo.most_common():
        print(f"     {label:10} {count:5}  {count / total:5.1%}")

    non_neutral = [(e, c) for e, c in emo.items() if e not in ("Flat", "Neutral")]
    top = max((c for _, c in non_neutral), default=0)
    verdict = "OK" if top / total < 0.55 else "OVER-FORCING (a non-neutral emotion dominates)"
    print(f"\n  Verdict: {verdict}")

    styled = [l for l in lines if (l.get("expression") or {}).get("expression_style")]
    if styled:
        print(f"\n  Character-aware delivery applied to {len(styled)} lines. Examples:")
        for l in styled[:5]:
            p = l.get("performance", {})
            ex = l.get("expression", {})
            print(f"     {ex.get('character'):12} {ex.get('emotion'):8} style={ex.get('expression_style'):11} "
                  f"speed={p.get('speed_modifier')} pitch={p.get('pitch_modifier')}")
    else:
        print("\n  (No character expression profiles present -- base emotion delivery only.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative emotion/performance evaluation")
    parser.add_argument("books", nargs="+", help="Corpus book name(s) under data/corpus/pipeline/")
    args = parser.parse_args()
    for book in args.books:
        evaluate(book)


if __name__ == "__main__":
    main()
