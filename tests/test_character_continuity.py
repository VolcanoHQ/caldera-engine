"""Character continuity: appearance index + a cross-scene DELIVERY modifier.

The guardrail tests are the point: continuity must scale only the magnitude of an
already-detected emotion's delivery. It can never change an emotion label, and it
literally cannot turn a flat line non-flat -- if it could, it would be the running-
history detection bias that once collapsed whole books onto one emotion.
"""

from src.character_continuity import (
    annotate_continuity,
    build_appearance_index,
    _decay,
    _CARRYOVER_BASE,
)


def _line(line_id, character, emotion, segment_type="dialogue", pitch=1.0, speed=1.0):
    return {
        "line_id": line_id, "character": character, "segment_type": segment_type,
        "text": "x", "emotion": emotion,
        "performance": {"pitch_modifier": pitch, "speed_modifier": speed,
                        "delivery_style": f"{emotion.lower()}_delivery"},
    }


def _scene(scene_id, lines):
    return {"scene_id": scene_id, "lines": lines}


def test_decay_is_full_when_consecutive_and_zero_past_max():
    assert _decay(1) == 1.0            # consecutive scenes
    assert _decay(5) < 1.0            # fading
    assert _decay(6) == 0.0          # past the window
    assert _decay(0) == 0.0
    assert _decay(None) == 0.0


def test_appearance_index_orders_scenes_and_dominant_emotion():
    scenes = [
        _scene("s1", [_line("l1", "Holmes", "Angry"), _line("l2", "Holmes", "Angry"), _line("l3", "Holmes", "Flat")]),
        _scene("s2", [_line("l4", "Watson", "Cheery")]),
        _scene("s3", [_line("l5", "Holmes", "Flat")]),
    ]
    idx = build_appearance_index(scenes)
    assert [a["scene_id"] for a in idx["Holmes"]] == ["s1", "s3"]
    assert idx["Holmes"][0]["emotion"] == "Angry"   # dominant non-flat in s1
    assert idx["Holmes"][1]["emotion"] == "Flat"    # nothing but flat in s3
    assert [a["position"] for a in idx["Watson"]] == [1]


def test_carryover_boosts_intensity_on_quick_reentry():
    scenes = [
        _scene("s1", [_line("l1", "Holmes", "Angry", pitch=1.10)]),   # activated
        _scene("s2", [_line("l2", "Holmes", "Angry", pitch=1.10)]),   # re-enters next scene
    ]
    annotate_continuity(scenes)
    reentry = scenes[1]["lines"][0]
    # deviation from 1.0 scaled up by (1 + carryover); carryover = base * decay(1) * 1
    expected = round(1.0 + 0.10 * (1.0 + _CARRYOVER_BASE), 4)
    assert reentry["performance"]["pitch_modifier"] == expected
    assert reentry["continuity"]["carryover"] == _CARRYOVER_BASE
    assert reentry["continuity"]["prev_emotion"] == "Angry"
    assert reentry["continuity"]["gap"] == 1


def test_first_appearance_has_no_carryover():
    scenes = [_scene("s1", [_line("l1", "Holmes", "Angry", pitch=1.10)])]
    annotate_continuity(scenes)
    line = scenes[0]["lines"][0]
    assert line["continuity"]["first_appearance"] is True
    assert line["performance"]["pitch_modifier"] == 1.10   # untouched


def test_no_carryover_from_a_flat_previous_scene():
    scenes = [
        _scene("s1", [_line("l1", "Holmes", "Flat", pitch=1.0)]),
        _scene("s2", [_line("l2", "Holmes", "Angry", pitch=1.10)]),
    ]
    annotate_continuity(scenes)
    line = scenes[1]["lines"][0]
    assert line["continuity"]["carryover"] == 0.0            # prev scene was flat -> no residual
    assert line["performance"]["pitch_modifier"] == 1.10


def test_carryover_decays_to_zero_past_the_window():
    scenes = [_scene("s1", [_line("l1", "Holmes", "Angry", pitch=1.10)])]
    for i in range(2, 9):   # Holmes absent for many scenes
        scenes.append(_scene(f"s{i}", [_line(f"n{i}", "Narrator", "Flat", segment_type="narrative")]))
    scenes.append(_scene("s9", [_line("l9", "Holmes", "Angry", pitch=1.10)]))  # returns far later
    annotate_continuity(scenes)
    late = scenes[-1]["lines"][0]
    assert late["continuity"]["carryover"] == 0.0
    assert late["performance"]["pitch_modifier"] == 1.10


def test_guardrail_flat_line_cannot_be_activated_by_continuity():
    """The core safety property: continuity scales deviation from 1.0, so a flat
    line (1.0 modifiers, zero deviation) stays exactly flat no matter the carryover."""
    scenes = [
        _scene("s1", [_line("l1", "Holmes", "Violent", pitch=1.15, speed=1.15)]),
        _scene("s2", [_line("l2", "Holmes", "Flat", pitch=1.0, speed=1.0)]),
    ]
    annotate_continuity(scenes)
    flat = scenes[1]["lines"][0]
    assert flat["performance"]["pitch_modifier"] == 1.0
    assert flat["performance"]["speed_modifier"] == 1.0
    assert flat["emotion"] == "Flat"                        # label never touched


def test_narrator_is_excluded_from_continuity():
    scenes = [
        _scene("s1", [_line("l1", "Holmes", "Angry", pitch=1.10)]),
        _scene("s2", [_line("n1", "Narrator", "Tense", segment_type="narrative", pitch=1.10)]),
    ]
    annotate_continuity(scenes)
    assert "continuity" not in scenes[1]["lines"][0]        # narration untouched


def test_continuity_never_changes_emotion_labels():
    scenes = [
        _scene("s1", [_line("l1", "Holmes", "Angry", pitch=1.10)]),
        _scene("s2", [_line("l2", "Holmes", "Sad", pitch=0.94)]),
    ]
    before = [l["emotion"] for s in scenes for l in s["lines"]]
    annotate_continuity(scenes)
    after = [l["emotion"] for s in scenes for l in s["lines"]]
    assert before == after
