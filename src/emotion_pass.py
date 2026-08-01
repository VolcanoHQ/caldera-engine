#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List, Optional, Tuple

EMOTION_PASS_VERSION = "emotion_pass_v1"

_EMOTION_LEXICON = {
    "cheery": {
        "happy", "happily", "cheerful", "cheery", "smile", "smiled", "laughed", "laughing",
        "joy", "joyful", "delight", "delighted", "glad", "warmly", "pleasantly",
    },
    "sad": {
        "sad", "sadly", "sorrow", "sorrowful", "grief", "grieved", "wept", "cry", "cried",
        "mournful", "tears", "tearful", "lament", "lonely", "despair",
    },
    "angry": {
        "angry", "anger", "furious", "rage", "raged", "snapped", "growled", "scowled",
        "irritated", "annoyed", "resentful",
    },
    "violent": {
        "violent", "violence", "attack", "attacked", "struck", "hit", "stabbed", "shot",
        "murder", "killed", "blood", "fight", "fought", "brawl", "assault", "threw", "slam",
    },
    "tense": {
        "tense", "tension", "nervous", "nervously", "uneasy", "uncertain", "suspense",
        "waited", "watchful", "quietly", "hushed", "hesitant", "grim",
    },
    "fearful": {
        "afraid", "fear", "fearful", "fright", "frightened", "terrified", "panic", "panicked",
        "dread", "horror", "startled", "shuddered",
    },
}

_EMOTION_ORDER = ("violent", "angry", "fearful", "sad", "tense", "cheery")
_EMOTION_TITLE = {
    "flat": "Flat",
    "neutral": "Neutral",
    "cheery": "Cheery",
    "sad": "Sad",
    "angry": "Angry",
    "violent": "Violent",
    "tense": "Tense",
    "fearful": "Fearful",
}

_DELIVERY_HINT_TO_EMOTION = {
    "whisper": "tense",
    "mutter": "sad",
    "shout": "angry",
    "scream": "fearful",
    "hiss": "angry",
    "snap": "angry",
}

# Public: the emotion vocabulary (excludes "flat", a per-line absence-of-signal
# state, not an emotion label the expression layer maps).
EMOTION_KEYS: Tuple[str, ...] = _EMOTION_ORDER + ("neutral",)
EMOTION_TITLES = set(_EMOTION_TITLE.values())
_EMOTION_KEY_BY_TITLE = {v: k for k, v in _EMOTION_TITLE.items()}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _lexicon_scores(text: str) -> Dict[str, float]:
    toks = _tokenize(text)
    total = max(1, len(toks))
    scores: Dict[str, float] = {}
    for emo, words in _EMOTION_LEXICON.items():
        hits = sum(1 for t in toks if t in words)
        scores[emo] = hits / total
    return scores


def _boost_from_punctuation(text: str, scores: Dict[str, float], *, segment_type: str) -> None:
    exclaim = text.count("!")
    question = text.count("?")
    if exclaim >= 2:
        scores["angry"] += 0.06
        scores["tense"] += 0.04
    elif exclaim == 1:
        scores["tense"] += 0.02
    if segment_type == "dialogue" and question >= 2:
        scores["tense"] += 0.03


def _dominant_emotion(scores: Dict[str, float]) -> Tuple[str, float]:
    best = "neutral"
    best_score = 0.0
    for emo in _EMOTION_ORDER:
        sc = scores.get(emo, 0.0)
        if sc > best_score:
            best = emo
            best_score = sc
    if best_score < 0.015:
        return ("flat", 0.55 if best_score == 0.0 else 0.58)
    return best, min(0.98, 0.60 + best_score * 6.0)


def enrich_scene_emotions(
    scene_id: str,
    lines: List[Dict[str, Any]],
    *,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Text-driven emotional classification over a scene; returns each line
    tagged with an ``emotion`` label plus scene metadata.

    Character-agnostic BY DESIGN: the same text yields the same emotion no
    matter who speaks. Character identity must never bias *classification* --
    doing so (static baselines + a running per-character history) caused whole
    books to collapse onto one emotion, since a prior alone cleared the
    neutral threshold and then self-reinforced. Character-specific *delivery*
    is applied downstream by the expression profile layer
    (src/expression_profile.py). See tests/test_emotion_distribution.py for the
    regression guardrails that lock this separation in.

    Signals used are all textual: the line's own lexicon + punctuation, light
    bleed from adjacent lines and the scene, and any ``attribution_delivery``
    hint the attribution reducer extracted from an expressive verb. Director
    corrections (``overrides``, keyed by line_id) win outright.
    """
    if not lines:
        return lines, {"scene_id": scene_id, "scene_emotion": "Neutral", "version": EMOTION_PASS_VERSION}

    scene_text = " ".join(str(l.get("text", "")) for l in lines)
    scene_scores = _lexicon_scores(scene_text)
    scene_emotion_key, scene_conf = _dominant_emotion(scene_scores)

    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        current = dict(line)
        text = str(current.get("text", ""))
        own_scores = _lexicon_scores(text)
        _boost_from_punctuation(text, own_scores, segment_type=str(current.get("segment_type", "")))

        if idx > 0:
            prev_scores = _lexicon_scores(str(lines[idx - 1].get("text", "")))
            for k, v in prev_scores.items():
                own_scores[k] += v * 0.18
        if idx + 1 < len(lines):
            next_scores = _lexicon_scores(str(lines[idx + 1].get("text", "")))
            for k, v in next_scores.items():
                own_scores[k] += v * 0.18

        for emo, v in scene_scores.items():
            own_scores[emo] += v * 0.22

        delivery = str(current.get("attribution_delivery") or "").strip().lower()
        if delivery in _DELIVERY_HINT_TO_EMOTION:
            own_scores[_DELIVERY_HINT_TO_EMOTION[delivery]] += 0.12

        emo_key, conf = _dominant_emotion(own_scores)

        line_id = str(current.get("line_id") or "")
        override = (overrides or {}).get(line_id)
        overridden = False
        if override and override.get("emotion"):
            override_title = str(override["emotion"]).strip().title()
            if override_title in EMOTION_TITLES:
                emotion_title = override_title
                emo_key = _EMOTION_KEY_BY_TITLE.get(override_title, emo_key)
                conf = 0.99
                overridden = True
            else:
                emotion_title = _EMOTION_TITLE.get(emo_key, "Neutral")
        else:
            emotion_title = _EMOTION_TITLE.get(emo_key, "Neutral")

        current["emotion"] = emotion_title
        current["emotion_context"] = {
            "scene_emotion": _EMOTION_TITLE.get(scene_emotion_key, "Neutral"),
            "line_emotion": emotion_title,
            "confidence": round(conf, 4),
            "scene_confidence": round(scene_conf, 4),
            "overridden": overridden,
            "version": EMOTION_PASS_VERSION,
        }
        out.append(current)

    return out, {
        "scene_id": scene_id,
        "scene_emotion": _EMOTION_TITLE.get(scene_emotion_key, "Neutral"),
        "scene_confidence": round(scene_conf, 4),
        "version": EMOTION_PASS_VERSION,
    }

