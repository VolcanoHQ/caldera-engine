#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List, Tuple

EMOTION_PASS_VERSION = "emotion_pass_v3"
PERFORMANCE_INTELLIGENCE_VERSION = "performance_intelligence_v1"
CHARACTER_PROFILE_MODEL_VERSION = "character_profile_v1"

NARRATOR_TONE_MODEL: Tuple[str, ...] = (
    "neutral",
    "reflective",
    "suspenseful",
    "somber",
    "urgent",
    "mysterious",
    "dramatic",
    "informative",
)

# Core emotional states (normalized keys)
_CORE_EMOTION_LEXICON = {
    "joy": {"joy", "joyful", "delight", "delighted", "grateful", "gratitude", "proud", "pride", "admire", "admiration"},
    "happiness": {"happy", "happily", "glad", "pleased", "smile", "smiled", "pleasant"},
    "excitement": {"excited", "excitement", "thrilled", "eager", "enthusiastic", "wonderful"},
    "affection": {"affection", "loving", "warm", "friendly", "supportive", "dear"},
    "anger": {"angry", "anger", "furious", "rage", "irritated", "annoyed", "resentful"},
    "frustration": {"frustrated", "frustration", "bothered", "exasperated"},
    "rage": {"raged", "enraged", "seething", "fuming"},
    "sadness": {"sad", "sadly", "sorrow", "sorrowful", "grief", "grieved", "regret", "regretful", "disappointed"},
    "grief": {"wept", "weep", "cry", "cried", "tears", "tearful", "heartbroken", "mournful", "despair"},
    "loneliness": {"lonely", "alone", "isolated", "abandoned"},
    "fear": {"fear", "afraid", "fright", "frightened", "terrified", "horror", "dread"},
    "anxiety": {"anxious", "anxiety", "nervous", "nervously", "uneasy", "apprehension", "hesitant"},
    "panic": {"panic", "panicked", "terror", "shaking"},
    "suspicion": {"suspicious", "suspicion", "skeptical", "doubtful", "uncertain"},
    "calm": {"calm", "steady", "composed", "reserved", "controlled", "detached"},
    "thoughtful": {"thoughtful", "reflective", "contemplative", "philosophical", "observant"},
    "neutral": set(),
}

_INTENT_LEXICON = {
    "asking": {"why", "what", "how", "when", "where", "who", "could", "would", "should", "?"},
    "informing": {"because", "therefore", "explain", "note", "remember", "tell", "know"},
    "persuading": {"please", "must", "should", "need", "convince", "trust", "believe"},
    "challenging": {"dare", "prove", "never", "wrong", "accuse", "defy"},
    "commanding": {"do", "stop", "go", "listen", "look", "come", "stand", "move", "hold"},
}

_RELATIONSHIP_LEXICON = {
    "friendly": {"friend", "dear", "thanks", "thank", "welcome"},
    "hostile": {"fool", "idiot", "hate", "despise", "damn", "curse"},
    "protective": {"protect", "safe", "careful", "watch", "guard"},
    "dominant": {"obey", "command", "order", "must", "now"},
    "submissive": {"sorry", "forgive", "please", "if you wish"},
    "respectful": {"sir", "madam", "lord", "lady"},
    "dismissive": {"whatever", "nonsense", "ridiculous", "absurd"},
}

_SCENE_CONTEXT_LEXICON = {
    "battle": {"battle", "war", "fight", "fought", "attack", "struck", "blood", "weapon"},
    "investigation": {"clue", "evidence", "case", "deduce", "mystery", "suspect", "question"},
    "romance": {"love", "kiss", "romance", "darling", "beloved", "affection"},
    "horror": {"dark", "shadow", "terror", "horror", "scream", "dread"},
    "action": {"run", "running", "chase", "jump", "dash", "urgent"},
    "drama": {"family", "betrayal", "conflict", "argument", "tears"},
    "comedy": {"laugh", "joke", "funny", "amused", "witty"},
}

_MODIFIER_HINTS = {
    "suppressed": {"quietly", "under his breath", "under her breath", "restrained", "composed"},
    "sarcastic": {"yeah right", "as if", "sure", "obviously"},
    "mocking": {"mocked", "taunted", "sneered"},
    "playful": {"playful", "teasing", "grin", "laughed"},
    "bitter": {"bitter", "resentful", "coldly"},
    "genuine": {"truly", "honestly", "sincerely"},
    "forced": {"forced", "reluctantly"},
}

_DELIVERY_HINT_TO_VOLUME = {
    "whisper": "whispering",
    "mutter": "quiet",
    "shout": "shouting",
    "scream": "shouting",
    "hiss": "firm",
    "snap": "forceful",
}

_EMOTION_TITLE = {
    "joy": "Joy",
    "happiness": "Happiness",
    "excitement": "Excitement",
    "affection": "Affection",
    "anger": "Angry",
    "frustration": "Frustration",
    "rage": "Rage",
    "sadness": "Sad",
    "grief": "Grief",
    "loneliness": "Loneliness",
    "fear": "Fear",
    "anxiety": "Anxiety",
    "panic": "Panic",
    "suspicion": "Suspicion",
    "calm": "Calm",
    "thoughtful": "Thoughtful",
    "neutral": "Neutral",
    "flat": "Flat",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _contains_phrase(text_l: str, phrase: str) -> bool:
    return phrase in text_l


def _score_from_lexicon(text: str, lexicon: Dict[str, set]) -> Dict[str, float]:
    toks = _tokenize(text)
    text_l = (text or "").lower()
    total = max(1, len(toks))
    scores: Dict[str, float] = {k: 0.0 for k in lexicon}
    for key, words in lexicon.items():
        hits = 0
        for w in words:
            if " " in w:
                if _contains_phrase(text_l, w):
                    hits += 2
            elif w in toks:
                hits += 1
        scores[key] = hits / total
    return scores


def _sorted_top_two(scores: Dict[str, float]) -> Tuple[str, float, str, float]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top1 = ranked[0] if ranked else ("neutral", 0.0)
    top2 = ranked[1] if len(ranked) > 1 else ("neutral", 0.0)
    return top1[0], top1[1], top2[0], top2[1]


def _scene_intensity(scene_scores: Dict[str, float], scene_text: str) -> float:
    exclaims = scene_text.count("!")
    intense = scene_scores.get("rage", 0.0) + scene_scores.get("panic", 0.0) + scene_scores.get("fear", 0.0)
    raw = 0.20 + intense * 2.6 + min(0.25, exclaims * 0.02)
    return max(0.0, min(1.0, raw))


def _derive_delivery_factors(
    *,
    text: str,
    segment_type: str,
    attribution_delivery: str,
    primary_emotion: str,
) -> Dict[str, Any]:
    text_l = text.lower()
    exclaims = text.count("!")
    words = max(1, len(_tokenize(text)))

    volume = "normal"
    if attribution_delivery in _DELIVERY_HINT_TO_VOLUME:
        volume = _DELIVERY_HINT_TO_VOLUME[attribution_delivery]
    elif exclaims >= 2 or primary_emotion in ("rage", "panic"):
        volume = "loud"
    elif "whisper" in text_l:
        volume = "whispering"

    if words <= 4 and exclaims >= 1:
        pace = "hurried"
    elif words >= 22:
        pace = "deliberate"
    else:
        pace = "conversational"
    if "breathless" in text_l or "out of breath" in text_l:
        pace = "breathless"

    if any(k in text_l for k in ("exhausted", "tired", "weary")):
        energy = "tired"
    elif any(k in text_l for k in ("energized", "excited", "urgent", "quickly")) or exclaims >= 1:
        energy = "energized"
    else:
        energy = "alert" if segment_type == "dialogue" else "relaxed"

    if any(k in text_l for k in ("restrained", "composed", "calmly", "quietly")):
        control = "composed"
    elif primary_emotion in ("rage", "panic") or exclaims >= 2:
        control = "agitated"
    elif primary_emotion in ("anger", "fear", "anxiety"):
        control = "restrained"
    else:
        control = "controlled"

    emphasis = []
    uppercase_words = re.findall(r"\b[A-Z]{2,}\b", text)
    if uppercase_words:
        emphasis.extend([w.lower() for w in uppercase_words[:3]])
    if exclaims:
        emphasis.append("exclamation_stress")
    if "?" in text:
        emphasis.append("interrogative_stress")

    return {
        "volume": volume,
        "energy": energy,
        "pace": pace,
        "control": control,
        "emphasis": emphasis[:4],
        "timing": {
            "pause_weight": 0.65 if pace in ("deliberate", "slow") else 0.35,
            "cadence": "staccato" if pace in ("hurried", "breathless") else "natural",
        },
    }


def _derive_intent(text: str, segment_type: str) -> str:
    scores = _score_from_lexicon(text, _INTENT_LEXICON)
    if "?" in text and segment_type == "dialogue":
        scores["asking"] += 0.25
    first = (_tokenize(text)[:1] or [""])[0]
    if first in {"do", "go", "stop", "listen", "come", "look", "stand"}:
        scores["commanding"] += 0.20
    intent, score, _, _ = _sorted_top_two(scores)
    return intent if score > 0.02 else "informing"


def _derive_relationship(text: str, segment_type: str) -> str:
    if segment_type != "dialogue":
        return "observant"
    scores = _score_from_lexicon(text, _RELATIONSHIP_LEXICON)
    rel, score, _, _ = _sorted_top_two(scores)
    return rel if score > 0.02 else "neutral"


def _derive_scene_context(scene_text: str) -> str:
    scores = _score_from_lexicon(scene_text, _SCENE_CONTEXT_LEXICON)
    ctx, score, _, _ = _sorted_top_two(scores)
    return ctx if score > 0.015 else "drama"


def _derive_modifiers(text: str) -> List[str]:
    text_l = text.lower()
    mods: List[str] = []
    for key, hints in _MODIFIER_HINTS.items():
        if any((_contains_phrase(text_l, h) if " " in h else h in _tokenize(text_l)) for h in hints):
            mods.append(key)
    if "!" in text and "suppressed" not in mods:
        mods.append("strong")
    if not mods:
        mods.append("genuine")
    return mods[:3]


def _derive_archetype_mode(character: str, text: str, scene_context: str) -> str:
    name = (character or "").lower()
    text_l = text.lower()
    if any(k in name for k in ("holmes", "detective")) or scene_context == "investigation":
        return "detective"
    if any(k in text_l for k in ("courage", "brave", "save", "protect")):
        return "hero"
    if any(k in text_l for k in ("threat", "obey", "dominate", "menacing")):
        return "villain"
    if any(k in text_l for k in ("lesson", "learn", "teach", "wise", "patience")):
        return "mentor"
    if any(k in text_l for k in ("joke", "laugh", "funny", "witty")):
        return "comic_relief"
    return "none"


def _derive_environmental_effects(text: str) -> List[str]:
    text_l = text.lower()
    effects = []
    if "out of breath" in text_l or "breathless" in text_l:
        effects.append("out_of_breath")
    if "crying" in text_l or "sob" in text_l or "wept" in text_l:
        effects.append("crying")
    if "laughing" in text_l or "laughed" in text_l:
        effects.append("laughing")
    if "running" in text_l or "ran" in text_l:
        effects.append("running")
    if "injured" in text_l or "wounded" in text_l:
        effects.append("injured")
    if "dying" in text_l:
        effects.append("dying")
    return effects


def _normalize_emotion_title(key: str) -> str:
    return _EMOTION_TITLE.get(key, "Neutral")


def _map_narrator_tone(*, primary_emotion: str, scene_context: str, intensity: float, intent: str) -> str:
    if intensity >= 0.8:
        return "dramatic"
    if scene_context in ("horror", "battle") and primary_emotion in ("fear", "panic", "anxiety", "suspicion"):
        return "suspenseful"
    if primary_emotion in ("grief", "sadness", "loneliness"):
        return "somber"
    if scene_context == "investigation" or primary_emotion in ("suspicion", "thoughtful"):
        return "mysterious"
    if primary_emotion in ("rage", "anger", "frustration") or intent in ("commanding", "challenging"):
        return "urgent"
    if intent == "informing":
        return "informative"
    if primary_emotion in ("thoughtful", "calm"):
        return "reflective"
    return "neutral"


def _performance_intelligence(
    *,
    line_id: str,
    character: str,
    segment_type: str,
    profile: Dict[str, Any],
    confidence: float,
) -> Dict[str, Any]:
    delivery = profile.get("delivery", {})
    result = {
        "line_id": line_id,
        "character": character,
        "primary_emotion": profile.get("primary_emotion", "neutral"),
        "secondary_emotion": profile.get("secondary_emotion", "neutral"),
        "intent": profile.get("intent", "informing"),
        "intensity": profile.get("intensity", 0.5),
        "delivery_style": delivery.get("control", "controlled"),
        "pacing": delivery.get("pace", "conversational"),
        "confidence": round(float(confidence), 4),
        "tier_strategy": {
            "tier1": "character_aware_narrator",
            "tier2": "character_voices",
            "tier3": "dramatic_production",
        },
        "version": PERFORMANCE_INTELLIGENCE_VERSION,
    }
    if segment_type == "narrative":
        result["narrator_tone"] = _map_narrator_tone(
            primary_emotion=str(profile.get("primary_emotion", "neutral")),
            scene_context=str(profile.get("scene_context", "drama")),
            intensity=float(profile.get("intensity", 0.5)),
            intent=str(profile.get("intent", "informing")),
        )
    return result


def _character_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "unknown"


def _accent_hint(character: str) -> str:
    low = (character or "").lower()
    if any(k in low for k in ("holmes", "watson", "sherlock")):
        return "british_gentleman"
    if any(k in low for k in ("frankenstein", "victor")):
        return "european_formal"
    return "neutral_standard"


def build_character_performance_profiles(
    scenes: List[Dict[str, Any]],
    *,
    overrides: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    aggregates: Dict[str, Dict[str, Any]] = {}

    for scene in scenes:
        for line in scene.get("lines", []):
            character = str(line.get("character") or "").strip()
            if not character or character == "Narrator":
                continue
            profile = dict(line.get("emotion_profile") or {})
            intelligence = dict(line.get("performance_intelligence") or {})
            if not profile:
                continue

            bucket = aggregates.setdefault(character, {
                "count": 0,
                "authority_sum": 0.0,
                "confidence_sum": 0.0,
                "intent_counts": {},
                "relationship_counts": {},
                "pace_counts": {},
                "energy_counts": {},
                "style_counts": {},
                "emotion_counts": {},
                "modifiers": {},
                "archetype_counts": {},
            })

            bucket["count"] += 1
            intent = str(profile.get("intent", "informing"))
            relationship = str(profile.get("relationship", "neutral"))
            pace = str((profile.get("delivery") or {}).get("pace", "conversational"))
            energy = str((profile.get("delivery") or {}).get("energy", "alert"))
            delivery_style = str(intelligence.get("delivery_style", "controlled"))
            primary = str(profile.get("primary_emotion", "neutral"))
            archetype = str(profile.get("archetype_mode", "none"))
            conf = float(intelligence.get("confidence", 0.65))

            authority = 0.52
            if intent in ("commanding", "challenging"):
                authority += 0.28
            if relationship in ("dominant", "hostile"):
                authority += 0.12
            if relationship in ("submissive",):
                authority -= 0.16
            authority = max(0.0, min(1.0, authority))

            bucket["authority_sum"] += authority
            bucket["confidence_sum"] += conf
            bucket["intent_counts"][intent] = bucket["intent_counts"].get(intent, 0) + 1
            bucket["relationship_counts"][relationship] = bucket["relationship_counts"].get(relationship, 0) + 1
            bucket["pace_counts"][pace] = bucket["pace_counts"].get(pace, 0) + 1
            bucket["energy_counts"][energy] = bucket["energy_counts"].get(energy, 0) + 1
            bucket["style_counts"][delivery_style] = bucket["style_counts"].get(delivery_style, 0) + 1
            bucket["emotion_counts"][primary] = bucket["emotion_counts"].get(primary, 0) + 1
            bucket["archetype_counts"][archetype] = bucket["archetype_counts"].get(archetype, 0) + 1
            for modifier in profile.get("modifiers", []):
                bucket["modifiers"][modifier] = bucket["modifiers"].get(modifier, 0) + 1

    def _top_key(counter: Dict[str, int], default: str) -> str:
        if not counter:
            return default
        return sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    profiles = []
    for character, bucket in sorted(aggregates.items(), key=lambda kv: kv[0].lower()):
        n = max(1, int(bucket["count"]))
        emotional = sorted(bucket["emotion_counts"].items(), key=lambda kv: kv[1], reverse=True)[:4]
        mods = sorted(bucket["modifiers"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        profile = {
            "character_id": _character_id(character),
            "character": character,
            "performance_profile": {
                "archetype": _top_key(bucket["archetype_counts"], "none"),
                "authority": round(float(bucket["authority_sum"]) / n, 4),
                "confidence": round(float(bucket["confidence_sum"]) / n, 4),
                "pace": _top_key(bucket["pace_counts"], "conversational"),
                "energy": _top_key(bucket["energy_counts"], "alert"),
                "delivery": _top_key(bucket["style_counts"], "controlled"),
                "speaking_style": _top_key(bucket["intent_counts"], "informing"),
                "accent": _accent_hint(character),
                "relationship_dynamics": _top_key(bucket["relationship_counts"], "neutral"),
                "emotional_tendencies": {k: v for k, v in emotional},
                "modifiers": [m for m, _ in mods] or ["genuine"],
            },
            "line_count": n,
            "version": CHARACTER_PROFILE_MODEL_VERSION,
        }
        override = (overrides or {}).get(profile["character_id"]) or (overrides or {}).get(character)
        if isinstance(override, dict):
            perf_override = override.get("performance_profile") if isinstance(override.get("performance_profile"), dict) else {}
            if perf_override:
                profile["performance_profile"].update(perf_override)
            if isinstance(override.get("note"), str) and override["note"].strip():
                profile["override_note"] = override["note"].strip()[:200]
        profiles.append(profile)
    return profiles


def enrich_scene_emotions(scene_id: str, lines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Layered NLP emotional pass over one scene."""
    if not lines:
        return lines, {
            "scene_id": scene_id,
            "scene_emotion": "Neutral",
            "scene_intensity": 0.0,
            "scene_context": "drama",
            "version": EMOTION_PASS_VERSION,
        }

    scene_text = " ".join(str(l.get("text", "")) for l in lines)
    scene_scores = _score_from_lexicon(scene_text, _CORE_EMOTION_LEXICON)
    scene_primary, scene_primary_score, scene_secondary, _ = _sorted_top_two(scene_scores)
    scene_intensity = _scene_intensity(scene_scores, scene_text)
    scene_context = _derive_scene_context(scene_text)

    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        current = dict(line)
        text = str(current.get("text", ""))
        seg_type = str(current.get("segment_type", "narrative"))

        own_scores = _score_from_lexicon(text, _CORE_EMOTION_LEXICON)
        if idx > 0:
            prev_scores = _score_from_lexicon(str(lines[idx - 1].get("text", "")), _CORE_EMOTION_LEXICON)
            for k, v in prev_scores.items():
                own_scores[k] += v * 0.15
        if idx + 1 < len(lines):
            next_scores = _score_from_lexicon(str(lines[idx + 1].get("text", "")), _CORE_EMOTION_LEXICON)
            for k, v in next_scores.items():
                own_scores[k] += v * 0.15
        for k, v in scene_scores.items():
            own_scores[k] += v * 0.20

        line_primary, line_primary_score, line_secondary, _ = _sorted_top_two(own_scores)
        if line_primary_score < 0.015:
            line_primary = "flat"
            line_secondary = scene_primary if scene_primary_score > 0.01 else "neutral"
        exclaim_boost = min(0.2, text.count("!") * 0.05)
        line_intensity = max(0.0, min(1.0, 0.25 + line_primary_score * 3.0 + exclaim_boost + scene_intensity * 0.25))

        attr_delivery = str(current.get("attribution_delivery", "")).lower().strip()
        delivery_factors = _derive_delivery_factors(
            text=text,
            segment_type=seg_type,
            attribution_delivery=attr_delivery,
            primary_emotion=line_primary,
        )
        intent = _derive_intent(text, seg_type)
        relationship = _derive_relationship(text, seg_type)
        modifiers = _derive_modifiers(text)
        archetype_mode = _derive_archetype_mode(str(current.get("character", "")), text, scene_context)
        env_effects = _derive_environmental_effects(text)
        speech_style = "narration" if seg_type == "narrative" else "dialogue"

        profile = {
            "primary_emotion": line_primary,
            "secondary_emotion": line_secondary,
            "intensity": round(line_intensity, 4),
            "delivery": {
                "volume": delivery_factors["volume"],
                "energy": delivery_factors["energy"],
                "pace": delivery_factors["pace"],
                "control": delivery_factors["control"],
                "timing": delivery_factors["timing"],
                "emphasis": delivery_factors["emphasis"],
            },
            "intent": intent,
            "relationship": relationship,
            "scene_context": scene_context,
            "modifiers": modifiers,
            "speech_style": speech_style,
            "archetype_mode": archetype_mode,
            "environmental_effects": env_effects,
            "scene_intensity": round(scene_intensity, 4),
            "camera_intensity": round(min(1.0, line_intensity * 0.85 + (0.1 if seg_type == "dialogue" else 0.0)), 4),
            "gesture_intensity": round(min(1.0, line_intensity * 0.9), 4),
            "facial_intensity": round(min(1.0, line_intensity * 0.8), 4),
            "version": EMOTION_PASS_VERSION,
        }
        if seg_type == "narrative":
            profile["narrator_tone"] = _map_narrator_tone(
                primary_emotion=line_primary,
                scene_context=scene_context,
                intensity=line_intensity,
                intent=intent,
            )

        current["emotion"] = _normalize_emotion_title(line_primary)
        perf = dict(current.get("performance") or {})
        if not perf:
            perf = {"pitch_modifier": 1.0, "speed_modifier": 1.0, "delivery_style": "neutral_narrative"}
        existing_style = str(perf.get("delivery_style", "neutral_narrative"))
        if existing_style in ("neutral_narrative", "descriptive", "", "flat"):
            perf["delivery_style"] = f"{line_primary}_delivery" if line_primary not in ("flat", "neutral") else "neutral_narrative"
        if profile["delivery"]["pace"] in ("hurried", "fast", "breathless"):
            perf["speed_modifier"] = max(float(perf.get("speed_modifier", 1.0)), 1.08)
        elif profile["delivery"]["pace"] in ("deliberate", "slow"):
            perf["speed_modifier"] = min(float(perf.get("speed_modifier", 1.0)), 0.95)
        if profile["delivery"]["volume"] in ("whispering", "quiet"):
            perf["pitch_modifier"] = min(float(perf.get("pitch_modifier", 1.0)), 0.98)
        elif profile["delivery"]["volume"] in ("loud", "shouting", "forceful"):
            perf["pitch_modifier"] = max(float(perf.get("pitch_modifier", 1.0)), 1.02)
        current["performance"] = perf

        confidence = round(min(0.98, 0.58 + line_primary_score * 2.8), 4)
        current["emotion_context"] = {
            "scene_emotion": _normalize_emotion_title(scene_primary),
            "line_emotion": _normalize_emotion_title(line_primary),
            "confidence": confidence,
            "scene_confidence": round(min(0.98, 0.58 + scene_primary_score * 2.6), 4),
            "version": EMOTION_PASS_VERSION,
        }
        current["emotion_profile"] = profile
        current["performance_intelligence"] = _performance_intelligence(
            line_id=str(current.get("line_id", "")),
            character=str(current.get("character", "")),
            segment_type=seg_type,
            profile=profile,
            confidence=confidence,
        )
        out.append(current)

    return out, {
        "scene_id": scene_id,
        "scene_emotion": _normalize_emotion_title(scene_primary),
        "scene_secondary_emotion": _normalize_emotion_title(scene_secondary),
        "scene_intensity": round(scene_intensity, 4),
        "scene_context": scene_context,
        "version": EMOTION_PASS_VERSION,
    }

