#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.emotion_pass import (
    CHARACTER_PROFILE_MODEL_VERSION,
    EMOTION_PASS_VERSION,
    NARRATOR_TONE_MODEL,
    PERFORMANCE_INTELLIGENCE_VERSION,
    build_character_performance_profiles,
    enrich_scene_emotions,
)
from src.models import ManuscriptManifest

ATTRIBUTION_REDUCTION_VERSION = "attribution_reduction_v1"

_NEUTRAL_VERBS = {
    "say", "said", "says", "reply", "replied", "respond", "responded",
    "answer", "answered", "ask", "asked",
}
_EXPRESSIVE_VERBS = {
    "whisper", "whispered",
    "shout", "shouted",
    "scream", "screamed",
    "mutter", "muttered",
    "hiss", "hissed",
    "snap", "snapped",
}
_PRONOUN_SUBJECTS = {"he", "she", "they", "i", "we", "you"}

_SUBJECT_RE = r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*|he|she|they|I|we|you)"
_VERB_RE = r"(?:say|said|says|reply|replied|respond|responded|answer|answered|ask|asked|whisper|whispered|shout|shouted|scream|screamed|mutter|muttered|hiss|hissed|snap|snapped)"

_ATTRIB_SUBJ_VERB = re.compile(
    rf"^\s*(?P<subject>{_SUBJECT_RE})\s+(?P<verb>{_VERB_RE})(?P<tail>[^.!?]*)[.!?;,:\s]*$",
    re.IGNORECASE,
)
_ATTRIB_VERB_SUBJ = re.compile(
    rf"^\s*(?P<verb>{_VERB_RE})\s+(?P<subject>{_SUBJECT_RE})(?P<tail>[^.!?]*)[.!?;,:\s]*$",
    re.IGNORECASE,
)

_DIALOGUE_TAG_TRAIL = re.compile(
    rf'^\s*(?P<quote>["“][^"”]+["”])\s*,?\s*(?P<tag>(?:(?:{_SUBJECT_RE})\s+(?:{_VERB_RE})|(?:{_VERB_RE})\s+(?:{_SUBJECT_RE}))(?:[^.!?]*)?)\s*[.!?]*\s*$',
    re.IGNORECASE,
)

_DELIVERY_CANON = {
    "whisper": "whisper",
    "whispered": "whisper",
    "shout": "shout",
    "shouted": "shout",
    "scream": "scream",
    "screamed": "scream",
    "mutter": "mutter",
    "muttered": "mutter",
    "hiss": "hiss",
    "hissed": "hiss",
    "snap": "snap",
    "snapped": "snap",
}


class AttributionReductionEntry(BaseModel):
    scene_id: str
    source_line_id: str
    speaker: Optional[str] = None
    attribution_text: str
    classification: Literal["required", "convertible", "redundant"]
    action: Literal["keep", "metadata", "remove"]
    confidence: float = Field(default=0.5)
    delivery: Optional[str] = None
    reason: str = ""


class PerformanceLine(BaseModel):
    scene_id: str
    source_line_id: str
    segment_type: Literal["dialogue", "narrative"]
    character: str
    text: str
    delivery: Optional[str] = None
    attribution_reduction: Optional[Dict[str, Any]] = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_subject(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.lower() in _PRONOUN_SUBJECTS:
        return s.lower()
    return s


def _parse_attribution_text(text: str) -> Optional[Tuple[str, str, str]]:
    for pat in (_ATTRIB_SUBJ_VERB, _ATTRIB_VERB_SUBJ):
        m = pat.match((text or "").strip())
        if not m:
            continue
        subject = _normalize_subject(m.group("subject"))
        verb = (m.group("verb") or "").strip().lower()
        tail = (m.group("tail") or "").strip(" ,;:-")
        return subject, verb, tail
    return None


def _is_ambiguous_pronoun(subject: str, prev_speaker: Optional[str], next_speaker: Optional[str]) -> bool:
    if subject not in _PRONOUN_SUBJECTS:
        return False
    if not prev_speaker and not next_speaker:
        return True
    if prev_speaker and next_speaker and prev_speaker != next_speaker:
        return True
    return False


def _infer_neighbor_dialogue_speakers(lines: List[Dict[str, Any]], idx: int) -> Tuple[Optional[str], Optional[str]]:
    prev_speaker = None
    next_speaker = None
    for i in range(idx - 1, -1, -1):
        if lines[i].get("segment_type") == "dialogue" and lines[i].get("character") and lines[i].get("character") != "Narrator":
            prev_speaker = lines[i].get("character")
            break
    for i in range(idx + 1, len(lines)):
        if lines[i].get("segment_type") == "dialogue" and lines[i].get("character") and lines[i].get("character") != "Narrator":
            next_speaker = lines[i].get("character")
            break
    return prev_speaker, next_speaker


def _classify_attribution(
    *,
    scene_id: str,
    line_id: str,
    attribution_text: str,
    subject: str,
    verb: str,
    tail: str,
    inferred_speaker: Optional[str],
    prev_speaker: Optional[str],
    next_speaker: Optional[str],
) -> AttributionReductionEntry:
    if _is_ambiguous_pronoun(subject, prev_speaker, next_speaker):
        return AttributionReductionEntry(
            scene_id=scene_id,
            source_line_id=line_id,
            speaker=inferred_speaker,
            attribution_text=attribution_text,
            classification="required",
            action="keep",
            confidence=0.95,
            reason="ambiguous_pronoun_speaker",
        )

    if verb in _EXPRESSIVE_VERBS:
        delivery = _DELIVERY_CANON.get(verb, verb)
        return AttributionReductionEntry(
            scene_id=scene_id,
            source_line_id=line_id,
            speaker=inferred_speaker,
            attribution_text=attribution_text,
            classification="convertible",
            action="metadata",
            delivery=delivery,
            confidence=0.95,
            reason="expressive_verb",
        )

    if tail:
        return AttributionReductionEntry(
            scene_id=scene_id,
            source_line_id=line_id,
            speaker=inferred_speaker,
            attribution_text=attribution_text,
            classification="required",
            action="keep",
            confidence=0.9,
            reason="contains_staging_or_emotion_tail",
        )

    if verb in _NEUTRAL_VERBS and inferred_speaker:
        return AttributionReductionEntry(
            scene_id=scene_id,
            source_line_id=line_id,
            speaker=inferred_speaker,
            attribution_text=attribution_text,
            classification="redundant",
            action="remove",
            confidence=0.93,
            reason="speaker_already_resolved",
        )

    return AttributionReductionEntry(
        scene_id=scene_id,
        source_line_id=line_id,
        speaker=inferred_speaker,
        attribution_text=attribution_text,
        classification="required",
        action="keep",
        confidence=0.7,
        reason="fallback_keep",
    )


def _apply_override(
    entry: AttributionReductionEntry,
    override: Dict[str, Any],
) -> AttributionReductionEntry:
    decision = (override or {}).get("decision")
    classification = (override or {}).get("classification")
    if classification in ("required", "convertible", "redundant"):
        entry.classification = classification
        if classification == "required":
            entry.action = "keep"
        elif classification == "convertible":
            entry.action = "metadata"
        else:
            entry.action = "remove"

    if decision == "restore_removed":
        entry.classification = "required"
        entry.action = "keep"
        entry.reason = "director_override_restore_removed"
    elif decision == "remove_preserved":
        entry.classification = "redundant"
        entry.action = "remove"
        entry.reason = "director_override_remove_preserved"
    elif decision == "change_classification":
        entry.reason = "director_override_change_classification"
    return entry


def reduce_scene_lines(
    scene_id: str,
    lines: List[Dict[str, Any]],
    *,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[AttributionReductionEntry]]:
    out_lines: List[Dict[str, Any]] = []
    reductions: List[AttributionReductionEntry] = []

    for idx, line in enumerate(lines):
        current = dict(line)
        text = str(current.get("text") or "").strip()
        if not text:
            out_lines.append(current)
            continue

        seg_type = current.get("segment_type")
        line_id = str(current.get("line_id") or f"{scene_id}:{idx}")
        prev_speaker, next_speaker = _infer_neighbor_dialogue_speakers(lines, idx)
        inferred_speaker = current.get("character")
        if not inferred_speaker or inferred_speaker == "Narrator":
            inferred_speaker = prev_speaker or next_speaker

        reduced = False

        override = (overrides or {}).get(line_id, {})

        if seg_type == "narrative":
            parsed = _parse_attribution_text(text)
            if parsed:
                subject, verb, tail = parsed
                entry = _classify_attribution(
                    scene_id=scene_id,
                    line_id=line_id,
                    attribution_text=text,
                    subject=subject,
                    verb=verb,
                    tail=tail,
                    inferred_speaker=inferred_speaker,
                    prev_speaker=prev_speaker,
                    next_speaker=next_speaker,
                )
                entry = _apply_override(entry, override)
                reductions.append(entry)
                if entry.action == "remove":
                    reduced = True
                elif entry.action == "metadata":
                    current["attribution_delivery"] = entry.delivery
                    current["attribution_reduction"] = entry.model_dump()

        elif seg_type == "dialogue":
            m = _DIALOGUE_TAG_TRAIL.match(text)
            if m:
                quote = (m.group("quote") or "").strip()
                tag = (m.group("tag") or "").strip()
                parsed = _parse_attribution_text(tag)
                if parsed:
                    subject, verb, tail = parsed
                    entry = _classify_attribution(
                        scene_id=scene_id,
                        line_id=line_id,
                        attribution_text=tag,
                        subject=subject,
                        verb=verb,
                        tail=tail,
                        inferred_speaker=inferred_speaker,
                        prev_speaker=prev_speaker,
                        next_speaker=next_speaker,
                    )
                    entry = _apply_override(entry, override)
                    reductions.append(entry)
                    if entry.action in ("remove", "metadata"):
                        current["text"] = quote.strip(" ,")
                        current["attribution_reduction"] = entry.model_dump()
                        if entry.action == "metadata":
                            current["attribution_delivery"] = entry.delivery

        if not reduced:
            out_lines.append(current)

    return out_lines, reductions


def build_performance_script(manifest_path: str) -> str:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())

    book_stem = os.path.splitext(os.path.basename(manifest.source_file))[0]
    tier3_dir = os.path.join("data", "corpus", "pipeline", book_stem, "tier3")
    os.makedirs(tier3_dir, exist_ok=True)
    override_path = os.path.join(tier3_dir, "attribution_reduction_overrides.json")
    all_overrides = {}
    if os.path.exists(override_path):
        try:
            with open(override_path, "r", encoding="utf-8") as f:
                all_overrides = json.load(f) or {}
        except Exception:
            all_overrides = {}
    profile_override_path = os.path.join(tier3_dir, "character_performance_profile_overrides.json")
    profile_overrides = {}
    if os.path.exists(profile_override_path):
        try:
            with open(profile_override_path, "r", encoding="utf-8") as f:
                profile_overrides = json.load(f) or {}
        except Exception:
            profile_overrides = {}

    scenes_out: List[Dict[str, Any]] = []
    all_reductions: List[Dict[str, Any]] = []
    scene_emotion_pass: List[Dict[str, Any]] = []

    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                lines = [l.model_dump() for l in scene.lines]
                reduced_lines, reductions = reduce_scene_lines(
                    scene.scene_id,
                    lines,
                    overrides=all_overrides.get(scene.scene_id, {}),
                )
                reduced_lines, scene_emotion_meta = enrich_scene_emotions(scene.scene_id, reduced_lines)
                all_reductions.extend([r.model_dump() for r in reductions])
                scene_emotion_pass.append(scene_emotion_meta)
                scenes_out.append({
                    "scene_id": scene.scene_id,
                    "lines": reduced_lines,
                })

    character_profiles = build_character_performance_profiles(
        scenes_out,
        overrides=profile_overrides,
    )

    payload = {
        "detector_version": ATTRIBUTION_REDUCTION_VERSION,
        "generated_at": _utcnow(),
        "book": book_stem,
        "scenes": scenes_out,
        "reductions": all_reductions,
        "summary": {
            "total_scenes": len(scenes_out),
            "total_reductions": len(all_reductions),
            "removed_count": sum(1 for r in all_reductions if r["action"] == "remove"),
            "metadata_count": sum(1 for r in all_reductions if r["action"] == "metadata"),
            "kept_count": sum(1 for r in all_reductions if r["action"] == "keep"),
        },
        "emotion_pass": {
            "version": EMOTION_PASS_VERSION,
            "scenes": scene_emotion_pass,
        },
        "performance_intelligence": {
            "version": PERFORMANCE_INTELLIGENCE_VERSION,
            "tier_model": {
                "tier1": "character_aware_narrator",
                "tier2": "character_voices",
                "tier3": "dramatic_production",
            },
            "narrator_tone_model": {
                "version": CHARACTER_PROFILE_MODEL_VERSION,
                "tones": list(NARRATOR_TONE_MODEL),
            },
        },
        "character_performance_profiles": character_profiles,
    }

    out_path = os.path.join(tier3_dir, "performance_script.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_performance_scene_lines(book: str, scene_id: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join("data", "corpus", "pipeline", book, "tier3", "performance_script.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for scene in data.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                return [dict(l) for l in scene.get("lines", [])]
    except Exception:
        return None
    return None
