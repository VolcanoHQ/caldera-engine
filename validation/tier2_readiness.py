#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tier 2 readiness check.

Tier 2 (narrator + attributed character voices) needs, from the analysis, for
every dialogue line: a REAL character attribution (drives voice selection), an
emotion (drives synthesis), and reduced text (no tags read aloud) -- plus a
castable roster. This confirms the performance script provides all of it, per
book, and prints a READY / NOT-READY verdict with the exact gaps.

Reads the artifacts the validation harness already built under validation/output/.

    python validation/tier2_readiness.py "A Scandal in Bohemia" "TheTaleofPeterRabbit"
"""

import json
import os
import sys

OUT_ROOT = os.path.join("validation", "output")
_ATTR_COVERAGE_GATE = 0.95      # share of dialogue that must be attributed to a real character


def _load(book, name):
    p = os.path.join(OUT_ROOT, book, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def assess(book):
    perf = _load(book, "performance_script.json")
    if perf is None:
        return {"book": book, "error": "no performance_script.json in validation/output/ (run validate_book.py first)"}
    consolidated = (_load(book, "character_profiles_consolidated.json") or {}).get("characters", [])

    lines = [l for s in perf["scenes"] for l in s["lines"]]
    dialogue = [l for l in lines if l.get("segment_type") == "dialogue"]
    attributed = [l for l in dialogue if l.get("character") and l["character"] != "Narrator"]
    roster = sorted({l["character"] for l in attributed})

    # per-requirement coverage
    coverage = (len(attributed) / len(dialogue)) if dialogue else 0.0
    emotion_ok = sum(1 for l in dialogue if l.get("emotion"))
    text_ok = sum(1 for l in dialogue if (l.get("text") or "").strip())
    reductions = perf.get("reductions", [])
    tags_silenced = sum(1 for r in reductions if r["action"] in ("remove", "metadata"))
    # every roster character castable = has a voice_affinity marketplace query
    prof_by_name = {p["identity"]["name"]: p for p in consolidated}
    castable = [c for c in roster if (prof_by_name.get(c, {}).get("voice_affinity", {}) or {}).get("marketplace_query")]

    gaps = []
    if dialogue and coverage < _ATTR_COVERAGE_GATE:
        narr = len(dialogue) - len(attributed)
        gaps.append(f"attribution: only {coverage:.0%} of dialogue attributed to a real character "
                    f"({narr}/{len(dialogue)} still Narrator) — needs LLM enrichment for Tier 2")
    if not roster:
        gaps.append("roster: no distinct speaking characters — cannot cast voices")
    if dialogue and emotion_ok < len(dialogue):
        gaps.append(f"emotion: {len(dialogue)-emotion_ok} dialogue lines missing an emotion")
    if dialogue and text_ok < len(dialogue):
        gaps.append(f"text: {len(dialogue)-text_ok} dialogue lines have empty text")
    if roster and len(castable) < len(roster):
        gaps.append(f"casting: {len(roster)-len(castable)} of {len(roster)} characters lack a voice_affinity query")

    return {
        "book": book,
        "ready": not gaps and bool(dialogue),
        "dialogue_lines": len(dialogue),
        "attributed": len(attributed),
        "attribution_coverage": round(coverage, 3),
        "roster": roster,
        "roster_size": len(roster),
        "castable": len(castable),
        "emotion_coverage": f"{emotion_ok}/{len(dialogue)}",
        "text_coverage": f"{text_ok}/{len(dialogue)}",
        "tags_silenced": tags_silenced,
        "gaps": gaps,
    }


if __name__ == "__main__":
    for book in sys.argv[1:]:
        a = assess(book)
        print("=" * 74)
        if a.get("error"):
            print(f"{book}: {a['error']}"); continue
        tag = "✅ TIER 2 READY" if a["ready"] else "⚠️ NOT READY"
        print(f"{tag}  —  {book}")
        print(f"  dialogue: {a['dialogue_lines']}  attributed: {a['attributed']} ({a['attribution_coverage']:.0%})  "
              f"roster: {a['roster_size']} ({a['castable']} castable)")
        print(f"  emotion: {a['emotion_coverage']}  text: {a['text_coverage']}  tags silenced: {a['tags_silenced']}")
        print(f"  roster: {a['roster']}")
        for g in a["gaps"]:
            print(f"    ⚠️ {g}")
