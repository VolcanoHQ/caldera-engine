#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audiobook validation harness — first-pass QA of Caldera Engine analysis.

For each book it (re)builds the understanding + performance-intelligence artifacts
from the existing tier1 ingest, then computes QA metrics and auto-flags red flags
that a human reviewer (second pass) should look at. Runs the zero-LLM layer only
(structure -> manifest -> performance script -> consolidated character profiles);
it does NOT re-run LLM attribution/G4/the Tier 3 crew.

    python validation/validate_book.py "TheTaleofPeterRabbit" "A Scandal in Bohemia"

Writes per book to validation/output/<book>/:
  performance_script.json, character_profiles_consolidated.json  (copied artifacts)
  metrics.json                                                    (machine metrics)
  SUMMARY.md                                                      (human-readable + flags)
"""

import collections
import json
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.book_structure_adapter import load_line_payloads, load_structure, structure_to_manifest
from src.attribution_reduction import build_performance_script
from src.character_profile import build_character_profiles

PIPELINE_ROOT = os.path.join("data", "corpus", "pipeline")
OUT_ROOT = os.path.join("validation", "output")
_NEUTRAL = {"Flat", "Neutral"}
_OVER_FORCING_CEILING = 0.55


def _raw_payload_line_count(book):
    for name in ("loop4_lines_enriched.json", "loop4_lines.json"):
        p = os.path.join(PIPELINE_ROOT, book, "tier1", name)
        if os.path.exists(p):
            data = json.load(open(p, encoding="utf-8"))
            return sum(len(s.get("lines", [])) for s in data), name
    return 0, None


def _metrics(book, perf, profiles, manifest_lines):
    # perf lines are POST attribution-reduction (redundant tags removed); the
    # manifest count is PRE-reduction. Real structural loss = raw payload vs
    # manifest; reduction removals are expected, not loss.
    lines = [l for s in perf["scenes"] for l in s["lines"]]
    dialogue = [l for l in lines if l.get("segment_type") == "dialogue"]
    raw_lines, raw_src = _raw_payload_line_count(book)

    speakers = collections.Counter(l.get("character") for l in dialogue)
    emo = collections.Counter(l.get("emotion") for l in lines)
    total = max(1, len(lines))
    non_neutral = {e: c for e, c in emo.items() if e not in _NEUTRAL}
    max_non_neutral = max(non_neutral.values(), default=0)

    styled = [l for l in lines if (l.get("expression") or {}).get("expression_style")]
    reductions = perf.get("reductions", [])
    roles = collections.Counter(p["identity"]["role"] for p in profiles)

    flags = []
    performance_lines = len(lines)                  # POST attribution-reduction
    dropped = raw_lines - manifest_lines            # REAL structural loss (0 = healthy)
    reduction_removed = manifest_lines - performance_lines  # expected, not loss
    if dropped > 0:
        flags.append(f"STRUCTURE: {dropped} of {raw_lines} payload lines missing from the built manifest")
    if len(perf["scenes"]) < 2 and raw_lines > 20:
        flags.append(f"STRUCTURE: only {len(perf['scenes'])} scene(s) for a {raw_lines}-line book — likely under-segmented")
    narr = speakers.get("Narrator", 0)
    if dialogue and narr == len(dialogue):
        flags.append("ATTRIBUTION: 100% of dialogue is 'Narrator' — no character attribution (needs enrichment)")
    if max_non_neutral / total >= _OVER_FORCING_CEILING:
        top = max(non_neutral, key=non_neutral.get)
        flags.append(f"EMOTION: over-forcing — '{top}' is {max_non_neutral/total:.0%} of lines (>{_OVER_FORCING_CEILING:.0%})")
    if not profiles:
        flags.append("PROFILES: no consolidated character profiles produced")

    return {
        "book": book,
        "structure": {
            "parts": None,  # filled by caller
            "scenes": len(perf["scenes"]),
            "manifest_lines": manifest_lines,
            "performance_lines": performance_lines,
            "raw_payload_lines": raw_lines,
            "raw_source": raw_src,
            "dropped_lines": dropped,
            "reduction_removed": reduction_removed,
        },
        "attribution": {
            "dialogue_lines": len(dialogue),
            "narration_lines": len(lines) - len(dialogue),
            "distinct_speakers": len([s for s in speakers if s and s != "Narrator"]),
            "speakers": dict(speakers.most_common()),
            "pct_narrator_of_dialogue": round(100 * narr / max(1, len(dialogue)), 1),
        },
        "attribution_reduction": {
            "total": len(reductions),
            "by_action": dict(collections.Counter(r["action"] for r in reductions)),
            "examples": [
                {"action": r["action"], "delivery": r.get("delivery"), "text": r["attribution_text"][:70], "reason": r["reason"]}
                for r in reductions[:6]
            ],
        },
        "emotion": {
            "distribution": {e: c for e, c in emo.most_common()},
            "distribution_pct": {e: round(100 * c / total, 1) for e, c in emo.most_common()},
            "max_non_neutral_pct": round(100 * max_non_neutral / total, 1),
            "over_forcing": max_non_neutral / total >= _OVER_FORCING_CEILING,
            "non_flat_examples": [
                {"emotion": l["emotion"], "character": l.get("character"), "text": l.get("text", "")[:60]}
                for l in lines if l.get("emotion") not in _NEUTRAL
            ][:10],
        },
        "expression": {
            "profiles_present": perf.get("expression_profile", {}).get("characters", []),
            "styled_lines": len(styled),
        },
        "continuity": {
            "lines_modified": perf.get("character_continuity", {}).get("lines_modified", 0),
            "characters_indexed": len(perf.get("character_index", {})),
        },
        "character_profiles": {
            "count": len(profiles),
            "roles": dict(roles),
            "arcs": [
                {"name": p["identity"]["name"], "role": p["identity"]["role"],
                 "appearances": len(p["arc"]["appearances"]), "dominant": p["arc"]["dominant_emotion"],
                 "range": p["arc"]["emotional_range"]}
                for p in sorted(profiles, key=lambda p: -len(p["arc"]["appearances"]))[:8]
            ],
        },
        "flags": flags,
    }


def _summary_md(m):
    s = m["structure"]; a = m["attribution"]; e = m["emotion"]; r = m["attribution_reduction"]
    cp = m["character_profiles"]
    L = []
    L.append(f"# Validation — {m['book']}\n")
    if m["flags"]:
        L.append("## ⚠️ Auto-flags for human review")
        for f in m["flags"]:
            L.append(f"- {f}")
    else:
        L.append("## ✅ No automated red flags")
    L.append("")
    L.append("## Structure")
    L.append(f"- scenes: **{s['scenes']}**  |  manifest lines: **{s['manifest_lines']}** / raw payload {s['raw_payload_lines']}"
             f"  |  dropped: **{s['dropped_lines']}**  |  reduced (tags removed): {s['reduction_removed']}")
    L.append("")
    L.append("## Attribution")
    L.append(f"- dialogue lines: {a['dialogue_lines']}  |  narration: {a['narration_lines']}  |  distinct speakers: {a['distinct_speakers']}")
    L.append(f"- Narrator share of dialogue: **{a['pct_narrator_of_dialogue']}%**")
    L.append(f"- speakers: {a['speakers']}")
    L.append("")
    L.append("## Attribution reduction")
    L.append(f"- total: {r['total']}  {r['by_action']}")
    for ex in r["examples"]:
        L.append(f"  - [{ex['action']}] {ex['delivery'] or ''} \"{ex['text']}\" ({ex['reason']})")
    L.append("")
    L.append("## Emotion")
    L.append(f"- distribution: {e['distribution_pct']}")
    L.append(f"- max non-neutral: {e['max_non_neutral_pct']}%  |  over-forcing: {e['over_forcing']}")
    for ex in e["non_flat_examples"]:
        L.append(f"  - [{ex['emotion']}] ({ex['character']}) \"{ex['text']}\"")
    L.append("")
    L.append("## Expression & continuity")
    L.append(f"- expression profiles present: {m['expression']['profiles_present']}  |  styled lines: {m['expression']['styled_lines']}")
    L.append(f"- continuity lines modified: {m['continuity']['lines_modified']}  |  characters indexed: {m['continuity']['characters_indexed']}")
    L.append("")
    L.append("## Character profiles (L7)")
    L.append(f"- count: {cp['count']}  roles: {cp['roles']}")
    for arc in cp["arcs"]:
        L.append(f"  - {arc['name']} [{arc['role']}] — {arc['appearances']} scenes, dominant {arc['dominant']}, range {arc['range']}")
    return "\n".join(L) + "\n"


def validate(book):
    out_dir = os.path.join(OUT_ROOT, book)
    os.makedirs(out_dir, exist_ok=True)
    try:
        structure = load_structure(book)
        manifest = structure_to_manifest(structure, line_payloads=load_line_payloads(book))
        tier3 = os.path.join(PIPELINE_ROOT, book, "tier3")
        os.makedirs(tier3, exist_ok=True)
        manifest_path = os.path.join(tier3, "canonical_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f)
        perf_path = build_performance_script(manifest_path)
        prof_path = build_character_profiles(book)

        perf = json.load(open(perf_path, encoding="utf-8"))
        profiles = json.load(open(prof_path, encoding="utf-8")).get("characters", []) if prof_path else []
        manifest_lines = sum(len(s.lines) for p in manifest.parts for c in p.chapters for s in c.scenes)

        m = _metrics(book, perf, profiles, manifest_lines)
        m["structure"]["parts"] = manifest.total_parts
        m["structure"]["chapters"] = manifest.total_chapters

        shutil.copy(perf_path, os.path.join(out_dir, "performance_script.json"))
        if prof_path:
            shutil.copy(prof_path, os.path.join(out_dir, "character_profiles_consolidated.json"))
        json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"), indent=2)
        open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8").write(_summary_md(m))
        return m
    except Exception as exc:
        err = {"book": book, "FAILED": str(exc), "trace": traceback.format_exc()}
        json.dump(err, open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"), indent=2)
        open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8").write(
            f"# Validation — {book}\n\n## ❌ PIPELINE FAILED\n\n```\n{exc}\n```\n")
        return err


if __name__ == "__main__":
    books = sys.argv[1:]
    for book in books:
        m = validate(book)
        if "FAILED" in m:
            print(f"❌ {book}: FAILED — {m['FAILED']}")
        else:
            flags = m["flags"]
            tag = "⚠️ " + f"{len(flags)} flag(s)" if flags else "✅ clean"
            print(f"{tag}  {book}: {m['structure']['scenes']} scenes, {m['structure']['manifest_lines']} lines, "
                  f"emotion max {m['emotion']['max_non_neutral_pct']}%, {m['character_profiles']['count']} profiles")
