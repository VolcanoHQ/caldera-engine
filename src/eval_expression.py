#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Expression-profile evaluation harness.

Scores the engine's authored character expression profiles
(`character_profiles.json → emotion_expression_profile`) against a human
`expression_gold.json` built per the annotation guidelines
(`docs/Caldera Engine Human Annotation Guidelines.md`, Dimension 5).

It answers: is Tier 1's character-aware narrator differentiating characters in a
way that matches human judgment? Metrics (all over the (character, emotion) pairs
present in BOTH sides):

  - exact agreement   -- same style string
  - band agreement    -- same band (reserved / balanced / intensified); a soft
                         match, since "restrained" vs "suppressed" is a near-miss
                         but "restrained" vs "explosive" is a real disagreement
  - coverage          -- gold-only / system-only / comparable pair counts
  - evidence support  -- fraction of gold entries carrying evidence (Rule 2)
  - calibration       -- mean gold confidence on matched vs unmatched pairs
                         (a calibrated annotator is more confident when right)

Both sides are canonicalized through the same vocabulary
(`canonicalize_expression_profile`), so noun emotions ("anger") and style
synonyms ("stoic") never cause a false mismatch.

    python -m src.eval_expression "A Scandal in Bohemia" --gold path/to/expression_gold.json
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.expression_profile import canonicalize_expression_profile, canonicalize_style, style_band


def _system_profiles_path(book: str) -> str:
    return os.path.join("data", "corpus", "pipeline", book, "tier3", "character_profiles.json")


def _canonical_system(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """{character: {emotion: style}} from character_profiles.json entries."""
    out: Dict[str, Dict[str, str]] = {}
    for item in profiles or []:
        if isinstance(item, dict) and item.get("name"):
            mapping = canonicalize_expression_profile(item.get("emotion_expression_profile"))
            if mapping:
                out[item["name"]] = mapping
    return out


def _canonical_gold(gold: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, Any]]]:
    """Return ({character: {emotion: style}}, {(character, emotion): meta}) where
    meta carries the gold entry's confidence/evidence. Accepts both the rich gold
    shape ({"style", "confidence", "evidence"}) and a bare {emotion: style} map."""
    styles: Dict[str, Dict[str, str]] = {}
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in gold or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("character")
        raw = entry.get("emotion_expression_profile") or {}
        if not name or not isinstance(raw, dict):
            continue
        # Flatten rich {emotion: {style,...}} to {emotion: style} for canonicalize,
        # keeping the per-entry meta for calibration/evidence scoring.
        flat: Dict[str, Any] = {}
        raw_meta: Dict[str, Dict[str, Any]] = {}
        for emotion, value in raw.items():
            if isinstance(value, dict):
                flat[emotion] = value.get("style")
                raw_meta[emotion] = value
            else:
                flat[emotion] = value
                raw_meta[emotion] = {}
        canon = canonicalize_expression_profile(flat)
        if not canon:
            continue
        styles[name] = canon
        # Re-key meta onto canonical emotion keys via the same canonicalization.
        for emotion, value in flat.items():
            ekey_map = canonicalize_expression_profile({emotion: value})
            for ekey in ekey_map:
                m = raw_meta.get(emotion, {})
                meta[(name, ekey)] = {
                    "confidence": m.get("confidence"),
                    "evidence": m.get("evidence") or [],
                    "inferred": m.get("inferred"),
                }
    return styles, meta


def score_expression(gold: List[Dict[str, Any]], system: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare a human expression gold against system character profiles. Pure;
    no I/O. Both inputs are lists of {name/character, emotion_expression_profile}."""
    gold_styles, gold_meta = _canonical_gold(gold)
    sys_styles = _canonical_system(system)

    gold_pairs = {(c, e) for c, m in gold_styles.items() for e in m}
    sys_pairs = {(c, e) for c, m in sys_styles.items() for e in m}
    comparable = sorted(gold_pairs & sys_pairs)

    exact = 0
    band = 0
    matched_conf: List[float] = []
    unmatched_conf: List[float] = []
    per_pair: List[Dict[str, Any]] = []
    for (char, emo) in comparable:
        g = gold_styles[char][emo]
        s = sys_styles[char][emo]
        is_exact = (g == s)
        is_band = (style_band(g) is not None and style_band(g) == style_band(s))
        exact += int(is_exact)
        band += int(is_band)
        conf = gold_meta.get((char, emo), {}).get("confidence")
        if isinstance(conf, (int, float)):
            (matched_conf if is_exact else unmatched_conf).append(float(conf))
        per_pair.append({"character": char, "emotion": emo, "gold": g, "system": s,
                         "exact": is_exact, "band": is_band})

    n = len(comparable)
    total_gold_entries = sum(len(m) for m in gold_styles.values())
    with_evidence = sum(1 for meta in gold_meta.values() if meta.get("evidence"))

    def _mean(xs: List[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "coverage": {
            "comparable_pairs": n,
            "gold_pairs": len(gold_pairs),
            "system_pairs": len(sys_pairs),
            "gold_only": sorted(gold_pairs - sys_pairs),
            "system_only": sorted(sys_pairs - gold_pairs),
            "characters_compared": sorted({c for c, _ in comparable}),
        },
        "agreement": {
            "exact": round(exact / n, 4) if n else None,
            "band": round(band / n, 4) if n else None,
            "exact_count": exact, "band_count": band, "n": n,
        },
        "evidence_support_rate": round(with_evidence / total_gold_entries, 4) if total_gold_entries else None,
        "calibration": {
            "mean_confidence_matched": _mean(matched_conf),
            "mean_confidence_unmatched": _mean(unmatched_conf),
        },
        "pairs": per_pair,
    }


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(book: str, gold_path: str) -> Dict[str, Any]:
    sys_path = _system_profiles_path(book)
    if not os.path.exists(sys_path):
        raise FileNotFoundError(
            f"No system character profiles at {sys_path} — run the character designer first "
            f"(console director refresh, or scene_director --design-characters-only).")
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"No expression gold at {gold_path}")
    return score_expression(_load_json(gold_path), _load_json(sys_path))


def _print_report(book: str, result: Dict[str, Any]) -> None:
    cov, agr, cal = result["coverage"], result["agreement"], result["calibration"]
    print("=" * 72)
    print(f"EXPRESSION EVAL: {book}")
    print("=" * 72)
    print(f"\nComparable (character, emotion) pairs: {cov['comparable_pairs']}"
          f"  (gold {cov['gold_pairs']}, system {cov['system_pairs']})")
    print(f"Characters compared: {', '.join(cov['characters_compared']) or '(none)'}")
    if agr["n"]:
        print(f"\nExact style agreement: {agr['exact']:.0%}  ({agr['exact_count']}/{agr['n']})")
        print(f"Band  agreement (reserved/balanced/intensified): {agr['band']:.0%}  ({agr['band_count']}/{agr['n']})")
    else:
        print("\nNo comparable pairs — cannot score agreement (check names/emotions overlap).")
    print(f"\nEvidence-support rate (gold entries w/ evidence): "
          f"{result['evidence_support_rate'] if result['evidence_support_rate'] is not None else 'n/a'}")
    print(f"Calibration — mean gold confidence: matched={cal['mean_confidence_matched']} "
          f"unmatched={cal['mean_confidence_unmatched']}")
    if cov["gold_only"]:
        print(f"\nGold-only pairs (system didn't profile): {cov['gold_only'][:8]}"
              + (" …" if len(cov["gold_only"]) > 8 else ""))
    if cov["system_only"]:
        print(f"System-only pairs (no human gold): {cov['system_only'][:8]}"
              + (" …" if len(cov["system_only"]) > 8 else ""))
    mism = [p for p in result["pairs"] if not p["exact"]]
    if mism:
        print("\nDisagreements:")
        for p in mism[:12]:
            tag = "band-ok" if p["band"] else "MISS"
            print(f"   {p['character']:16} {p['emotion']:8} gold={p['gold']:11} system={p['system']:11} [{tag}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score system expression profiles vs human gold")
    parser.add_argument("book", help="Corpus book name under data/corpus/pipeline/")
    parser.add_argument("--gold", required=True, help="Path to expression_gold.json")
    parser.add_argument("--json", action="store_true", help="Emit the raw result as JSON")
    args = parser.parse_args()
    result = evaluate(args.book, args.gold)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(args.book, result)


if __name__ == "__main__":
    main()
