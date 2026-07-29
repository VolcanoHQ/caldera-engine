#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Expression Profile layer.

Turns a *detected* emotion (from the text-only emotion pass) plus the speaking
character's expression style into concrete delivery: performance modifiers and a
delivery_style label. This is the ONLY place character identity is allowed to
shape a performance -- emotion *classification* stays character-agnostic
(src/emotion_pass.py). Two characters can be angry in the same scene; this layer
is what makes Holmes's restrained anger sound different from an explosive
character's, without either one changing which emotion was detected.

Deterministic and cheap (no LLM calls). All modifiers are absolute multipliers
centered on 1.0 (never signed deltas), matching every downstream consumer
(VoiceSynthesizer / production_mixer read performance.{pitch,speed}_modifier).
"""

from typing import Any, Dict, List, Optional

EXPRESSION_PROFILE_VERSION = "expression_profile_v1"

# Base per-emotion prosody as multipliers centered on 1.0. The expression style
# scales each modifier's *deviation from 1.0*, so "restrained" damps toward 1.0
# and "explosive" amplifies away from it -- the label never flips.
_EMOTION_PERFORMANCE: Dict[str, Dict[str, float]] = {
    "angry":   {"pitch_modifier": 1.05, "speed_modifier": 1.08},
    "violent": {"pitch_modifier": 1.06, "speed_modifier": 1.10},
    "fearful": {"pitch_modifier": 1.10, "speed_modifier": 1.06},
    "sad":     {"pitch_modifier": 0.94, "speed_modifier": 0.90},
    "tense":   {"pitch_modifier": 1.02, "speed_modifier": 0.96},
    "cheery":  {"pitch_modifier": 1.04, "speed_modifier": 1.05},
}
_NEUTRAL_PERFORMANCE = {"pitch_modifier": 1.0, "speed_modifier": 1.0}

# How strongly a character externalizes an emotion. 1.0 = the base prosody as-is;
# below 1.0 pulls delivery toward neutral, above 1.0 pushes it further.
_EXPRESSION_INTENSITY: Dict[str, float] = {
    "hidden": 0.35, "internalized": 0.4, "suppressed": 0.45, "restrained": 0.5,
    "subtle": 0.6, "dry": 0.6, "quiet": 0.6, "controlled": 0.7, "measured": 0.75,
    "neutral": 1.0,
    "open": 1.2, "expressive": 1.3, "overt": 1.35, "intense": 1.4, "explosive": 1.5,
}
_DEFAULT_INTENSITY = 1.0

# Detection emits Title-case labels; profiles and this layer key on lowercase.
_TITLE_TO_KEY = {
    "Cheery": "cheery", "Sad": "sad", "Angry": "angry", "Violent": "violent",
    "Tense": "tense", "Fearful": "fearful", "Flat": "flat", "Neutral": "neutral",
}

# One canonical emotion vocabulary. LLM-authored profiles often use the noun
# ("anger", "fear", "joy"); fold those onto the detection keys so a profile and
# the detector always speak the same language (taxonomy consolidation).
_EMOTION_ALIASES = {
    "anger": "angry", "rage": "angry", "irritation": "angry",
    "fear": "fearful", "fright": "fearful", "terror": "fearful", "dread": "fearful",
    "joy": "cheery", "happiness": "cheery", "happy": "cheery", "humor": "cheery", "amusement": "cheery",
    "sadness": "sad", "grief": "sad", "sorrow": "sad",
    "tension": "tense", "suspense": "tense", "nervousness": "tense",
    "violence": "violent",
}

# The recognized expression styles are exactly the intensity table's keys -- one
# source of truth for what the designer may emit and what this layer will honor.
EXPRESSION_STYLES = frozenset(_EXPRESSION_INTENSITY)

# Styles worth suggesting to the character designer (excludes the no-op "neutral").
SUGGESTED_EXPRESSION_STYLES = tuple(
    s for s in sorted(_EXPRESSION_INTENSITY, key=lambda k: _EXPRESSION_INTENSITY[k])
    if s != "neutral"
)

# Coarse grouping of styles into three bands, from most internalized to most
# externalized. Used for "near-miss" agreement scoring against human gold
# (restrained vs suppressed is the same band -- a soft match -- while restrained
# vs explosive is a real disagreement) and referenced by the annotation
# guidelines. Single source of truth for the grouping.
EXPRESSION_BANDS: Dict[str, tuple] = {
    "reserved": ("hidden", "internalized", "suppressed", "restrained", "subtle", "dry", "quiet"),
    "balanced": ("controlled", "measured"),
    "intensified": ("open", "expressive", "overt", "intense", "explosive"),
}
_STYLE_TO_BAND = {style: band for band, styles in EXPRESSION_BANDS.items() for style in styles}


def style_band(style: Any) -> Optional[str]:
    """The band (reserved / balanced / intensified) a style belongs to, or None
    for an unrecognized style or the no-op 'neutral'."""
    return _STYLE_TO_BAND.get(str(style or "").strip().lower())

# Common near-synonyms an LLM reaches for, folded onto the canonical styles so a
# reasonable answer is honored rather than silently degrading to no effect.
_STYLE_ALIASES = {
    "reserved": "restrained", "guarded": "restrained", "stoic": "restrained", "clipped": "restrained",
    "repressed": "suppressed", "bottled": "suppressed",
    "buried": "hidden", "concealed": "hidden", "masked": "hidden",
    "understated": "subtle", "faint": "subtle",
    "wry": "dry", "sardonic": "dry", "deadpan": "dry", "sarcastic": "dry",
    "calm": "controlled", "composed": "controlled", "cool": "controlled",
    "even": "measured", "steady": "measured", "deliberate": "measured",
    "soft": "quiet", "hushed": "quiet",
    "loud": "open", "outspoken": "open", "frank": "open", "direct": "open",
    "demonstrative": "expressive", "animated": "expressive", "effusive": "expressive",
    "theatrical": "overt", "dramatic": "overt", "showy": "overt",
    "fierce": "intense", "fervent": "intense", "passionate": "intense",
    "volcanic": "explosive", "eruptive": "explosive", "unrestrained": "explosive",
}


def _emotion_key(line: Dict[str, Any]) -> str:
    return _TITLE_TO_KEY.get(str(line.get("emotion") or "Neutral"), "neutral")


def _canonical_emotion(key: str) -> str:
    key = (key or "").strip().lower()
    return _EMOTION_ALIASES.get(key, key)


def canonicalize_style(style: Any) -> Optional[str]:
    """Map an LLM-authored expression style onto the canonical vocabulary via
    alias table; return None if it is not a recognized (or aliasable) style."""
    s = str(style or "").strip().lower()
    s = _STYLE_ALIASES.get(s, s)
    return s if s in EXPRESSION_STYLES else None


def canonicalize_expression_profile(raw: Any) -> Dict[str, str]:
    """Clean one character's ``emotion_expression_profile`` into a canonical,
    actionable ``{emotion_key: style}`` map: emotion keys folded onto the
    detection vocabulary, styles validated/aliased, and any entry that is not a
    real deliverable emotion with a recognized style dropped. Accepts arbitrary
    junk (non-dicts, nested values) and returns ``{}`` rather than raising -- the
    hardening boundary for character-designer output."""
    clean: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return clean
    for emotion, style in raw.items():
        ekey = _canonical_emotion(str(emotion))
        skey = canonicalize_style(style)
        if ekey in _EMOTION_PERFORMANCE and skey:
            clean[ekey] = skey
    return clean


def _clamp(x: float) -> float:
    return round(max(0.5, min(1.6, x)), 4)


def apply_expression_profile(
    lines: List[Dict[str, Any]],
    profiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Set each line's delivery (performance + delivery_style) from its detected
    emotion, personalized by the speaking character's expression style when a
    profile exists.

    ``profiles`` maps ``character -> {emotion_key: expression_style}`` (as built
    by :func:`load_expression_profiles`). A line with no matching profile entry
    still gets the base emotion prosody -- the profile only tunes *how much*.
    """
    profiles = profiles or {}
    out: List[Dict[str, Any]] = []
    for line in lines:
        current = dict(line)
        emo = _emotion_key(current)
        perf = dict(current.get("performance") or {})

        character = str(current.get("character") or "")
        is_dialogue = current.get("segment_type") == "dialogue"
        style = profiles.get(character, {}).get(emo) if is_dialogue else None
        intensity = _EXPRESSION_INTENSITY.get((style or "").lower(), _DEFAULT_INTENSITY)

        if emo in ("flat", "neutral"):
            perf["pitch_modifier"] = 1.0
            perf["speed_modifier"] = 1.0
            perf["delivery_style"] = "neutral_narrative"
        else:
            base = _EMOTION_PERFORMANCE[emo]
            perf["pitch_modifier"] = _clamp(1.0 + (base["pitch_modifier"] - 1.0) * intensity)
            perf["speed_modifier"] = _clamp(1.0 + (base["speed_modifier"] - 1.0) * intensity)
            perf["delivery_style"] = f"{emo}_{style}" if style else f"{emo}_delivery"

        current["performance"] = perf
        current["expression"] = {
            "character": character or None,
            "emotion": emo,
            "expression_style": style,
            "intensity": intensity,
            "version": EXPRESSION_PROFILE_VERSION,
        }
        out.append(current)
    return out


def load_expression_profiles(
    character_profiles: Optional[List[Dict[str, Any]]],
) -> Dict[str, Dict[str, str]]:
    """Extract ``{character: {emotion_key: style}}`` from character_profiles.json
    entries' ``emotion_expression_profile`` field. Emotion keys are canonicalized
    onto the detection vocabulary; unknown emotions are dropped. Missing or
    malformed profiles yield an empty mapping (delivery falls back to base)."""
    profiles: Dict[str, Dict[str, str]] = {}
    for item in character_profiles or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        mapping = canonicalize_expression_profile(item.get("emotion_expression_profile"))
        if name and mapping:
            profiles[name] = mapping
    return profiles
