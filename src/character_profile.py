#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""L7 — the consolidated Character Profile.

Assembles the per-character signals that today live scattered across
`character_profiles.json` (visual + expression) and `performance_script.json`
(appearance index + per-scene emotion) into ONE versioned, portable entity per
character. Deterministic (no LLM): it only reshapes data other loops already
produced, plus a stable schema and provenance.

This is the keystone the character-analysis loop and the future character
database/product hang off. It defaults to private (`consent.shareable = false`);
nothing here shares anything -- see
`docs/Caldera Engine Enrichment Loops & Character Intelligence.md`.
"""

import collections
import json
import os
import re
from typing import Any, Dict, List, Optional

from src.expression_profile import canonicalize_expression_profile

CHARACTER_PROFILE_VERSION = "character_profile_v1"

_NEUTRAL = {"Flat", "Neutral"}
_PRIMARY_MIN_APPEARANCES = 3   # scenes-spoken-in to count as a primary character
_PRIMARY_MIN_MENTIONS = 8      # name occurrences across the book to count as primary
_SUPPORTING_MIN_MENTIONS = 3


def _role(appearance_count: int, mention_count: int = 0) -> str:
    """Importance from narrative presence, not just dialogue volume: a character
    who is central but rarely speaks (e.g. Peter, one spoken line but mentioned
    throughout) should still rank primary."""
    if appearance_count >= _PRIMARY_MIN_APPEARANCES or mention_count >= _PRIMARY_MIN_MENTIONS:
        return "primary"
    if appearance_count >= 1 or mention_count >= _SUPPORTING_MIN_MENTIONS:
        return "supporting"
    return "mentioned"          # has a designed profile but neither speaks nor is named much


def count_mentions(names: List[str], texts: List[str]) -> Dict[str, int]:
    """How often each character is named across the book's line texts. Matches on
    the last name token (so "Mrs. Rabbit" and "Rabbit" both count), mirroring the
    character-designer's importance signal (scene_director.design_characters)."""
    full_text = "\n".join(texts)
    counts: Dict[str, int] = {}
    for name in names:
        token = (name or "").split()[-1] if name and name.split() else ""
        if not token:
            counts[name] = 0
            continue
        counts[name] = len(re.findall(r"\b" + re.escape(token) + r"\b", full_text))
    return counts


def _arc(appearances: List[Dict[str, Any]]) -> Dict[str, Any]:
    emotions = [a.get("emotion", "Flat") for a in appearances]
    non_flat = [e for e in emotions if e not in _NEUTRAL]
    dominant = collections.Counter(non_flat).most_common(1)[0][0] if non_flat else "Flat"
    ordered_range = sorted(set(emotions))
    return {
        "appearances": appearances,
        "dominant_emotion": dominant,
        "emotional_range": ordered_range,
    }


def _voice_query(name: str, visual_description: str) -> str:
    desc = (visual_description or "").strip()
    return f"{name}: {desc}" if desc else f"{name} audiobook character voice"


def consolidate_profiles(
    character_index: Dict[str, List[Dict[str, Any]]],
    designer_profiles: Optional[List[Dict[str, Any]]] = None,
    *,
    book_id: str,
    source_work: str,
    source_hash: Optional[str] = None,
    mention_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Build consolidated profiles from the appearance index (speaking presence
    per scene) and the designer's character_profiles.json (visual + expression).
    Pure; no I/O. Characters come from either source -- a character can have a
    designed visual profile but never speak (role 'mentioned'), or speak without
    a designed profile (visual empty). ``mention_counts`` (name occurrences across
    the book) lets a central-but-quiet character still rank primary."""
    mention_counts = mention_counts or {}
    designer_by_name = {
        item["name"]: item
        for item in (designer_profiles or [])
        if isinstance(item, dict) and item.get("name")
    }
    names = sorted(set(character_index) | set(designer_by_name))

    profiles: List[Dict[str, Any]] = []
    for name in names:
        appearances = character_index.get(name, [])
        mentions = int(mention_counts.get(name, 0))
        designer = designer_by_name.get(name, {})
        visual_description = str(designer.get("visual_description") or "")
        profiles.append({
            "profile_version": CHARACTER_PROFILE_VERSION,
            "identity": {
                "name": name,
                "aliases": list(designer.get("aliases") or []),
                "role": _role(len(appearances), mentions),
            },
            "presence": {
                "speaking_scenes": len(appearances),
                "mentions": mentions,
            },
            "provenance": {
                "book_id": book_id,
                "source_work": source_work,
                "source_hash": source_hash,
                "designed_by": designer.get("designed_by"),
            },
            "visual": {
                "visual_description": visual_description,
                "evidence": list(designer.get("evidence_snippets") or []),
                "inferred": bool(designer.get("inferred", not visual_description)),
            },
            "expression": canonicalize_expression_profile(designer.get("emotion_expression_profile")),
            "arc": _arc(appearances),
            "voice_affinity": {
                "marketplace_query": _voice_query(name, visual_description),
                "current_voice_ref": None,
            },
            "consent": {"shareable": False, "license": None},
        })
    return profiles


def _tier3_dir(book: str) -> str:
    return os.path.join("data", "corpus", "pipeline", book, "tier3")


def build_character_profiles(book: str) -> Optional[str]:
    """Read this book's performance script + designed profiles, write the
    consolidated `character_profiles_consolidated.json`. Returns the path, or
    None if the performance script hasn't been built yet."""
    tier3 = _tier3_dir(book)
    perf_path = os.path.join(tier3, "performance_script.json")
    if not os.path.exists(perf_path):
        return None
    with open(perf_path, "r", encoding="utf-8") as f:
        perf = json.load(f)
    character_index = perf.get("character_index", {})

    designer_profiles: List[Dict[str, Any]] = []
    designer_path = os.path.join(tier3, "character_profiles.json")
    if os.path.exists(designer_path):
        with open(designer_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                designer_profiles = loaded

    # Count how often each character is named across the whole book so a
    # central-but-quiet character isn't mis-ranked by speaking volume alone.
    names = sorted(set(character_index) | {p["name"] for p in designer_profiles if isinstance(p, dict) and p.get("name")})
    texts = [l.get("text", "") for s in perf.get("scenes", []) for l in s.get("lines", [])]
    mention_counts = count_mentions(names, texts)

    profiles = consolidate_profiles(
        character_index,
        designer_profiles,
        book_id=book,
        source_work=perf.get("book", book),
        mention_counts=mention_counts,
    )
    out_path = os.path.join(tier3, "character_profiles_consolidated.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"profile_version": CHARACTER_PROFILE_VERSION, "characters": profiles}, f, indent=2)
    os.replace(tmp, out_path)
    return out_path
