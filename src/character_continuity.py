#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Character continuity — the first scene/cast enrichment loop.

Two deterministic, zero-LLM products over an already-attributed, already-emotion-
tagged book:

  1. An **appearance index**: for each character, the ordered scenes they speak in
     and the dominant emotion they carried in each. Pure data enhancement -- the
     substrate future cast/relationship enrichment builds on.

  2. A **continuity entry-state modifier**: when a character re-enters a scene soon
     after an emotionally-activated one, residual state modulates HOW their current
     lines are delivered (intensity), decaying with the gap since they last spoke.

CRITICAL GUARDRAIL -- this lives strictly on the DELIVERY side, never detection.
It scales the *magnitude* of a line's already-computed performance modifiers; it
can never change which emotion was detected, and because it scales the deviation
from 1.0 it literally cannot turn a flat line non-flat (0 deviation stays 0). This
is the same detection/delivery boundary that the emotion pass enforces -- a
carryover that biased *classification* would recreate the running-history snowball
that once collapsed whole books onto one emotion.
"""

import collections
from typing import Any, Dict, List, Optional

CONTINUITY_VERSION = "character_continuity_v1"

# Max intensity boost from an immediate (gap=1) residual emotional state.
_CARRYOVER_BASE = 0.15
# Scenes after which residual state has fully faded.
_MAX_GAP = 5
# Same clamp band as the expression layer, so combined delivery stays bounded.
_CLAMP_LO, _CLAMP_HI = 0.5, 1.6

_NEUTRAL_EMOTIONS = {"Flat", "Neutral"}


def _clamp(x: float) -> float:
    return round(max(_CLAMP_LO, min(_CLAMP_HI, x)), 4)


def _decay(gap: Optional[int]) -> float:
    """Recency weight: gap=1 (consecutive scenes) -> 1.0, fading to 0 past _MAX_GAP."""
    if gap is None or gap <= 0 or gap > _MAX_GAP:
        return 0.0
    return round(1.0 - (gap - 1) / _MAX_GAP, 4)


def _dominant_emotions_by_character(lines: List[Dict[str, Any]]) -> Dict[str, str]:
    """Each speaking (non-Narrator dialogue) character's dominant emotion in a
    scene: the most common non-flat label among their lines, else 'Flat'."""
    by_char: Dict[str, List[str]] = {}
    for line in lines:
        if line.get("segment_type") != "dialogue":
            continue
        char = str(line.get("character") or "")
        if not char or char == "Narrator":
            continue
        by_char.setdefault(char, []).append(str(line.get("emotion") or "Flat"))
    out: Dict[str, str] = {}
    for char, emotions in by_char.items():
        non_flat = [e for e in emotions if e not in _NEUTRAL_EMOTIONS]
        out[char] = collections.Counter(non_flat).most_common(1)[0][0] if non_flat else "Flat"
    return out


def build_appearance_index(scenes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """{character: [{scene_id, position, emotion}, ...]} in scene order."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for position, scene in enumerate(scenes):
        scene_id = scene.get("scene_id", "")
        for char, emotion in _dominant_emotions_by_character(scene.get("lines", [])).items():
            index.setdefault(char, []).append(
                {"scene_id": scene_id, "position": position, "emotion": emotion})
    return index


def annotate_continuity(scenes: List[Dict[str, Any]]) -> int:
    """In-place: attach a `continuity` block to each speaking character's dialogue
    lines and, when residual state carries over, scale that line's performance
    modifiers' deviation from 1.0 by (1 + carryover). Returns lines modified.

    Carryover only exists when the character's PREVIOUS scene was emotionally
    activated (non-flat); it decays with the scene gap and is clamped. A flat
    current line is untouched (its 1.0 modifiers have no deviation to scale)."""
    last: Dict[str, tuple] = {}  # character -> (position, dominant_emotion)
    modified = 0

    for position, scene in enumerate(scenes):
        for line in scene.get("lines", []):
            if line.get("segment_type") != "dialogue":
                continue
            char = str(line.get("character") or "")
            if not char or char == "Narrator":
                continue

            prev = last.get(char)
            if prev is None:
                line["continuity"] = {
                    "first_appearance": True, "gap": None, "prev_scene": None,
                    "prev_emotion": None, "carryover": 0.0, "version": CONTINUITY_VERSION,
                }
                continue

            prev_pos, prev_emotion = prev
            gap = position - prev_pos
            activation = 0.0 if prev_emotion in _NEUTRAL_EMOTIONS else 1.0
            carryover = round(_CARRYOVER_BASE * _decay(gap) * activation, 4)

            if carryover > 0:
                perf = dict(line.get("performance") or {})
                changed = False
                for key in ("pitch_modifier", "speed_modifier"):
                    if key in perf:
                        new = _clamp(1.0 + (perf[key] - 1.0) * (1.0 + carryover))
                        if new != perf[key]:      # flat lines (0 deviation) are a no-op
                            perf[key] = new
                            changed = True
                if changed:
                    line["performance"] = perf
                    modified += 1

            line["continuity"] = {
                "first_appearance": False,
                "gap": gap,
                "prev_scene": scenes[prev_pos].get("scene_id"),
                "prev_emotion": prev_emotion,
                "carryover": carryover,
                "version": CONTINUITY_VERSION,
            }

        # Record each character's state for THIS scene only after the scene is
        # processed, so continuity is strictly cross-scene (never within-scene).
        for char, emotion in _dominant_emotions_by_character(scene.get("lines", [])).items():
            last[char] = (position, emotion)

    return modified
