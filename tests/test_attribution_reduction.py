import json
import os
import shutil
from pathlib import Path

import pytest

from src.attribution_reduction import (
    build_performance_script,
    reduce_scene_lines,
)
from src.console_api import save_attribution_reduction_override, save_emotion_override
from src.feedback_events import load_feedback_events


def _line(line_id: str, character: str, segment_type: str, text: str) -> dict:
    return {
        "line_id": line_id,
        "chapter": 1,
        "scene": 1,
        "line_number": 1,
        "character": character,
        "speaker_id": f"char_{character.lower()}",
        "segment_type": segment_type,
        "text": text,
        "emotion": "Neutral",
        "performance": {"pitch_modifier": 1.0, "speed_modifier": 1.0, "delivery_style": "neutral_narrative"},
        "post_padding_ms": 250,
        "attribution_method": "Tier 1 Default",
        "confidence": 1.0,
        "speaker_locked": True,
        "utterance_type": "speech",
    }


def test_simple_dialogue_redundant_attribution_removed():
    lines = [
        _line("l1", "Holmes", "dialogue", '"I agree."'),
        _line("l2", "Narrator", "narrative", "Holmes said."),
        _line("l3", "Watson", "dialogue", '"Excellent."'),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert not any(l["line_id"] == "l2" for l in reduced)
    assert any(r.source_line_id == "l2" and r.action == "remove" for r in reductions)


def test_honorific_titled_speaker_tag_is_removed():
    """A titled speaker with a benign temporal tail -- 'said old Mrs. Rabbit one
    morning' -- is a redundant tag between resolved dialogue and must be removed.
    (Before: the period in 'Mrs.' broke the subject regex, so it was never seen.)"""
    lines = [
        _line("l1", "Mrs. Rabbit", "dialogue", '"Now my dears,"'),
        _line("l2", "Narrator", "narrative", "said old Mrs. Rabbit one morning,"),
        _line("l3", "Mrs. Rabbit", "dialogue", '"you may go into the fields."'),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert not any(l["line_id"] == "l2" for l in reduced)
    assert any(r.source_line_id == "l2" and r.action == "remove" for r in reductions)


def test_staging_tail_is_preserved_not_swallowed():
    """A manner/staging tail must land in the tail and force a keep -- it must not
    be swallowed into the name (case-sensitive name tokens) and silently removed."""
    lines = [
        _line("l1", "Holmes", "dialogue", '"Come."'),
        _line("l2", "Narrator", "narrative", "said Mr. Holmes with a grim smile."),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert any(l["line_id"] == "l2" for l in reduced)          # kept, not removed
    entry = next(r for r in reductions if r.source_line_id == "l2")
    assert entry.action == "keep"
    assert entry.reason == "contains_staging_or_emotion_tail"


def test_titled_expressive_verb_becomes_delivery_metadata():
    lines = [
        _line("l1", "Mr. McGregor", "dialogue", '"Come here!"'),
        _line("l2", "Narrator", "narrative", "whispered Mr. McGregor."),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    entry = next(r for r in reductions if r.source_line_id == "l2")
    assert entry.action == "metadata" and entry.delivery == "whisper"


def test_ordinary_narration_with_a_title_is_not_a_false_positive():
    """A titled subject without an attribution verb is not an attribution tag."""
    lines = [
        _line("l1", "Narrator", "narrative", "Old Mrs. Rabbit took a basket and her umbrella."),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert any(l["line_id"] == "l1" for l in reduced)
    assert reductions == []


def test_pronoun_attribution_kept_when_ambiguous():
    lines = [
        _line("l1", "Holmes", "dialogue", '"I agree."'),
        _line("l2", "Narrator", "narrative", "he said."),
        _line("l3", "Watson", "dialogue", '"I disagree."'),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert any(l["line_id"] == "l2" for l in reduced)
    assert any(r.source_line_id == "l2" and r.classification == "required" for r in reductions)


def test_convertible_emotional_attribution_keeps_delivery_metadata():
    lines = [
        _line("l1", "Holmes", "dialogue", '"I agree."'),
        _line("l2", "Narrator", "narrative", "Holmes whispered."),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    target = next(l for l in reduced if l["line_id"] == "l2")
    assert target.get("attribution_delivery") == "whisper"
    assert any(r.source_line_id == "l2" and r.action == "metadata" for r in reductions)


def test_action_heavy_attribution_is_required():
    lines = [
        _line("l1", "Holmes", "dialogue", '"Come now."'),
        _line("l2", "Narrator", "narrative", "Holmes asked while studying the letter."),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert any(l["line_id"] == "l2" for l in reduced)
    assert any(r.source_line_id == "l2" and r.action == "keep" for r in reductions)


def test_dialogue_inline_sherlock_tag_is_removed_from_text():
    lines = [
        _line("l1", "Holmes", "dialogue", '"Do you see anything, Watson?" asked Holmes.'),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert reduced[0]["text"] == '"Do you see anything, Watson?"'
    assert any(r.source_line_id == "l1" and r.action in ("remove", "metadata") for r in reductions)


def test_epistolary_narration_remains_unchanged():
    lines = [
        _line("l1", "Narrator", "narrative", "Your affectionate brother, R. Walton."),
        _line("l2", "Narrator", "narrative", "Archangel, 28th March, 17—."),
    ]
    reduced, reductions = reduce_scene_lines("s1", lines)
    assert len(reduced) == 2
    assert reductions == []


def test_no_loss_of_speaker_identity_in_dialogue_lines():
    lines = [
        _line("l1", "Holmes", "dialogue", '"I agree."'),
        _line("l2", "Narrator", "narrative", "Holmes said."),
        _line("l3", "Watson", "dialogue", '"Excellent."'),
    ]
    reduced, _ = reduce_scene_lines("s1", lines)
    by_id = {l["line_id"]: l for l in reduced}
    assert by_id["l1"]["character"] == "Holmes"
    assert by_id["l3"]["character"] == "Watson"


def test_build_performance_script_writes_artifact():
    stem = "test_attr_reduction_fixture"
    pipeline_dir = Path("data") / "corpus" / "pipeline" / stem / "tier3"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / "canonical_manifest.json"
    manifest = {
        "source_file": f"{stem}.txt",
        "total_parts": 1,
        "total_chapters": 1,
        "total_scenes": 1,
        "parts": [{
            "part_id": "part_p1",
            "title": "Part 1",
            "chapters": [{
                "chapter_id": "part_p1_c1",
                "title": "Chapter 1",
                "scenes": [{
                    "scene_id": "part_p1_c1_s1",
                    "lines": [
                        _line("l1", "Holmes", "dialogue", '"I agree."'),
                        _line("l2", "Narrator", "narrative", "Holmes said."),
                    ],
                }],
            }],
        }],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    try:
        out_path = build_performance_script(str(manifest_path))
        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["summary"]["total_scenes"] == 1
        assert payload["summary"]["total_reductions"] >= 1
    finally:
        shutil.rmtree(Path("data") / "corpus" / "pipeline" / stem, ignore_errors=True)


def test_build_performance_script_applies_expression_profile():
    """The character's expression profile tunes DELIVERY of an already-detected
    emotion; it must not change the emotion label. Two characters, same angry
    line, different expression style -> same emotion, different prosody."""
    stem = "test_expression_profile_fixture"
    pipeline_dir = Path("data") / "corpus" / "pipeline" / stem / "tier3"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / "canonical_manifest.json"
    angry_text = "Furious and full of rage, he snapped with anger."
    manifest = {
        "source_file": f"{stem}.txt",
        "total_parts": 1, "total_chapters": 1, "total_scenes": 1,
        "parts": [{
            "part_id": "part_p1", "title": "Part 1",
            "chapters": [{
                "chapter_id": "part_p1_c1", "title": "Chapter 1",
                "scenes": [{
                    "scene_id": "part_p1_c1_s1",
                    "lines": [
                        _line("l1", "Holmes", "dialogue", angry_text),
                        _line("l2", "Brute", "dialogue", angry_text),
                    ],
                }],
            }],
        }],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(pipeline_dir / "character_profiles.json", "w", encoding="utf-8") as f:
        json.dump([
            {"name": "Holmes", "visual_description": "x",
             "emotion_expression_profile": {"angry": "restrained"}},
            {"name": "Brute", "visual_description": "x",
             "emotion_expression_profile": {"angry": "explosive"}},
        ], f)

    try:
        out_path = build_performance_script(str(manifest_path))
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        by_id = {l["line_id"]: l for l in payload["scenes"][0]["lines"]}
        holmes, brute = by_id["l1"], by_id["l2"]

        # Same emotion label for both -- classification is character-agnostic.
        assert holmes["emotion"] == brute["emotion"] == "Angry"
        # Different delivery: restrained pulls toward neutral, explosive pushes away.
        assert holmes["performance"]["speed_modifier"] < brute["performance"]["speed_modifier"]
        assert holmes["expression"]["expression_style"] == "restrained"
        assert brute["expression"]["expression_style"] == "explosive"
        assert payload["expression_profile"]["characters"] == ["Brute", "Holmes"]
    finally:
        shutil.rmtree(Path("data") / "corpus" / "pipeline" / stem, ignore_errors=True)


def test_emotion_override_records_feedback_event_and_persists(tmp_path, monkeypatch):
    import src.feedback_events as fe
    import src.console_api as capi
    monkeypatch.setattr(fe, "FEEDBACK_DIR", str(tmp_path / "feedback"))

    class _FakeStructure:
        structure_version = "test_v1"

    monkeypatch.setattr(capi, "load_structure", lambda book: _FakeStructure())

    stem = "test_emotion_override_fixture"
    pipeline_dir = Path("data") / "corpus" / "pipeline" / stem / "tier3"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / "canonical_manifest.json"
    manifest = {
        "source_file": f"{stem}.txt",
        "total_parts": 1,
        "total_chapters": 1,
        "total_scenes": 1,
        "parts": [{
            "part_id": "part_p1",
            "title": "Part 1",
            "chapters": [{
                "chapter_id": "part_p1_c1",
                "title": "Chapter 1",
                "scenes": [{
                    "scene_id": "part_p1_c1_s1",
                    "lines": [
                        _line("l1", "Grimm", "dialogue", "It is late."),
                    ],
                }],
            }],
        }],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    try:
        build_performance_script(str(manifest_path))

        result = save_emotion_override(
            stem,
            "part_p1_c1_s1",
            "l1",
            emotion="cheery",
            note="Grimm is actually delighted here.",
        )
        assert result == {"scene_id": "part_p1_c1_s1", "line_id": "l1", "emotion": "Cheery"}

        with open(pipeline_dir / "performance_script.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        line = payload["scenes"][0]["lines"][0]
        assert line["emotion"] == "Cheery"
        assert line["emotion_context"]["overridden"] is True

        events = load_feedback_events(stem)
        assert any(e.get("event_type") == "emotion_corrected" for e in events)
    finally:
        shutil.rmtree(Path("data") / "corpus" / "pipeline" / stem, ignore_errors=True)


def test_director_override_records_feedback_event(tmp_path, monkeypatch):
    import src.feedback_events as fe
    monkeypatch.setattr(fe, "FEEDBACK_DIR", str(tmp_path / "feedback"))

    book = "A Scandal in Bohemia"
    tier3_dir = Path("data") / "corpus" / "pipeline" / book / "tier3"
    tier3_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tier3_dir / "canonical_manifest.json"

    # Build performance script if missing from existing artifacts
    if not manifest_path.exists():
        pytest.skip("Canonical manifest missing for integration fixture")
    out_path = build_performance_script(str(manifest_path))
    with open(out_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not payload.get("reductions"):
        pytest.skip("No attribution reductions detected in fixture book")
    item = payload["reductions"][0]

    result = save_attribution_reduction_override(
        book,
        item["scene_id"],
        item["source_line_id"],
        decision="change_classification",
        classification="required",
        note="Director kept this for clarity.",
    )
    assert result is not None
    events = load_feedback_events(book)
    assert any(e.get("event_type") == "attribution_reduction_corrected" for e in events)

