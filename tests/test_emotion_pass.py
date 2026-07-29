from src.emotion_pass import enrich_scene_emotions


def _line(line_id: str, character: str, segment_type: str, text: str, delivery: str = "") -> dict:
    payload = {
        "line_id": line_id,
        "character": character,
        "segment_type": segment_type,
        "text": text,
        "emotion": "Neutral",
        "performance": {"pitch_modifier": 1.0, "speed_modifier": 1.0, "delivery_style": "neutral_narrative"},
    }
    if delivery:
        payload["attribution_delivery"] = delivery
    return payload


def test_scene_context_marks_tense_scene():
    lines = [
        _line("l1", "Narrator", "narrative", "The room was dark and everyone waited in silence."),
        _line("l2", "Holmes", "dialogue", "Did you hear that?"),
        _line("l3", "Watson", "dialogue", "Something is wrong!"),
    ]
    out, meta = enrich_scene_emotions("s1", lines)
    assert meta["scene_emotion"] in ("Tense", "Fearful")
    assert all("emotion_context" in l for l in out)


def test_happy_and_cheery_dialogue_gets_cheery_emotion():
    lines = [
        _line("l1", "Holmes", "dialogue", "Wonderful! I am delighted and happy to see you."),
    ]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion"] == "Cheery"


def test_sad_line_gets_sad_emotion():
    lines = [
        _line("l1", "Narrator", "narrative", "She wept in sorrow and despair."),
    ]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion"] == "Sad"


def test_angry_or_violent_line_detected():
    lines = [
        _line("l1", "Narrator", "narrative", "He attacked and struck with furious rage."),
    ]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion"] in ("Violent", "Angry")


def test_delivery_hint_influences_emotion():
    lines = [
        _line("l1", "Holmes", "narrative", "Holmes said.", delivery="whisper"),
    ]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion"] in ("Tense", "Fearful", "Sad")


def test_flat_fallback_when_no_signal():
    lines = [
        _line("l1", "Narrator", "narrative", "The table is by the window."),
    ]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion"] in ("Flat", "Neutral")


def test_detection_is_character_agnostic():
    """A neutral line stays flat regardless of who speaks: detection must never
    be biased by character identity (the removed baseline/history mechanism)."""
    line_a = _line("l1", "Holmes", "dialogue", "It is late.")
    line_b = _line("l1", "Moriarty", "dialogue", "It is late.")
    out_a, _ = enrich_scene_emotions("s1", [line_a])
    out_b, _ = enrich_scene_emotions("s1", [line_b])
    assert out_a[0]["emotion"] == out_b[0]["emotion"]
    assert out_a[0]["emotion"] in ("Flat", "Neutral")


def test_enrich_no_longer_accepts_character_params():
    """The character-bias parameters are gone from the detector's signature."""
    import inspect
    params = set(inspect.signature(enrich_scene_emotions).parameters)
    assert "character_baselines" not in params
    assert "character_history" not in params


def test_override_forces_emotion_and_marks_context():
    lines = [
        _line("l1", "Narrator", "narrative", "The table is by the window."),
    ]
    out, _ = enrich_scene_emotions("s1", lines, overrides={"l1": {"emotion": "Angry"}})
    assert out[0]["emotion"] == "Angry"
    assert out[0]["emotion_context"]["overridden"] is True


def test_unknown_override_emotion_is_ignored():
    lines = [
        _line("l1", "Narrator", "narrative", "The table is by the window."),
    ]
    out, _ = enrich_scene_emotions("s1", lines, overrides={"l1": {"emotion": "Bogus"}})
    assert out[0]["emotion"] in ("Flat", "Neutral")
    assert out[0]["emotion_context"]["overridden"] is False

