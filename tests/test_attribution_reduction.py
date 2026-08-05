import json
import os
import shutil
from pathlib import Path

import pytest

from src.attribution_reduction import (
    build_performance_script,
    reduce_scene_lines,
)
from src.console_api import (
    get_character_performance_profiles,
    save_attribution_reduction_override,
)
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
        assert "character_performance_profiles" in payload
        assert isinstance(payload["character_performance_profiles"], list)
        assert payload["performance_intelligence"]["tier_model"]["tier1"] == "character_aware_narrator"
        first_line = payload["scenes"][0]["lines"][0]
        assert "performance_intelligence" in first_line
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


def test_character_profile_override_records_feedback_event(tmp_path, monkeypatch):
    import src.feedback_events as fe
    from src.console_api import save_character_performance_profile_override

    monkeypatch.setattr(fe, "FEEDBACK_DIR", str(tmp_path / "feedback"))

    book = "A Scandal in Bohemia"
    tier3_dir = Path("data") / "corpus" / "pipeline" / book / "tier3"
    manifest_path = tier3_dir / "canonical_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Canonical manifest missing for integration fixture")
    build_performance_script(str(manifest_path))
    result = save_character_performance_profile_override(
        book,
        "Holmes",
        performance_profile={"authority": 0.91, "pace": "measured", "energy": "controlled"},
        note="Director tuned Holmes delivery for Tier 1 narration.",
    )
    assert result is not None
    events = load_feedback_events(book)
    assert any(e.get("event_type") == "character_performance_profile_corrected" for e in events)


def test_profile_api_returns_and_persists_snapshot():
    stem = "test_profile_api_fixture"
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
                        _line("l1", "Holmes", "dialogue", '"We should leave now."'),
                        _line("l2", "Watson", "dialogue", '"Are you certain?"'),
                    ],
                }],
            }],
        }],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    try:
        build_performance_script(str(manifest_path))
        payload = get_character_performance_profiles(stem)
        assert payload is not None
        assert payload["profile_count"] >= 2
        assert os.path.exists(payload["snapshot"]["snapshot_path"])
        assert os.path.exists(payload["snapshot"]["memory_log_path"])
    finally:
        shutil.rmtree(Path("data") / "corpus" / "pipeline" / stem, ignore_errors=True)


def test_profile_api_can_rebuild_if_missing():
    stem = "test_profile_api_refresh_fixture"
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
                    "lines": [_line("l1", "Holmes", "dialogue", '"Proceed."')],
                }],
            }],
        }],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    try:
        build_performance_script(str(manifest_path))
        os.remove(pipeline_dir / "performance_script.json")
        payload = get_character_performance_profiles(stem, refresh_if_missing=True)
        assert payload is not None
        assert payload["profile_count"] >= 1
        assert os.path.exists(pipeline_dir / "performance_script.json")
    finally:
        shutil.rmtree(Path("data") / "corpus" / "pipeline" / stem, ignore_errors=True)
