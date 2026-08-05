from src.emotion_pass import build_character_performance_profiles, enrich_scene_emotions


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


def test_scene_context_marks_tense_or_horror_scene():
    lines = [
        _line("l1", "Narrator", "narrative", "The room was dark and everyone waited in silence."),
        _line("l2", "Holmes", "dialogue", "Did you hear that?"),
        _line("l3", "Watson", "dialogue", "Something is wrong!"),
    ]
    out, meta = enrich_scene_emotions("s1", lines)
    assert meta["scene_context"] in ("horror", "drama")
    assert all("emotion_context" in l for l in out)
    assert all("emotion_profile" in l for l in out)
    assert all("performance_intelligence" in l for l in out)


def test_happy_dialogue_gets_positive_primary_emotion():
    lines = [_line("l1", "Holmes", "dialogue", "Wonderful! I am delighted and happy to see you.")]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion_profile"]["primary_emotion"] in ("happiness", "joy", "excitement", "affection")
    assert out[0]["emotion"] in ("Happiness", "Joy", "Excitement", "Affection")


def test_sad_line_gets_sadness_or_grief():
    lines = [_line("l1", "Narrator", "narrative", "She wept in sorrow and despair.")]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion_profile"]["primary_emotion"] in ("sadness", "grief")
    assert out[0]["emotion"] in ("Sad", "Grief")


def test_delivery_hint_maps_to_volume_and_control():
    lines = [_line("l1", "Holmes", "dialogue", "Holmes said we should stay quiet.", delivery="whisper")]
    out, _ = enrich_scene_emotions("s1", lines)
    profile = out[0]["emotion_profile"]
    assert profile["delivery"]["volume"] == "whispering"
    assert profile["delivery"]["control"] in ("composed", "restrained", "controlled")


def test_profile_includes_intent_relationship_and_modifiers():
    lines = [
        _line("l1", "Watson", "dialogue", "Please listen, sir, we must go now!"),
    ]
    out, _ = enrich_scene_emotions("s1", lines)
    profile = out[0]["emotion_profile"]
    assert profile["intent"] in ("commanding", "persuading", "informing")
    assert profile["relationship"] in ("respectful", "dominant", "friendly", "neutral")
    assert isinstance(profile["modifiers"], list) and profile["modifiers"]


def test_narrative_line_uses_constrained_narrator_tone():
    lines = [_line("l1", "Narrator", "narrative", "In the dark hall, he paused and listened.")]
    out, _ = enrich_scene_emotions("s1", lines)
    tone = out[0]["emotion_profile"].get("narrator_tone")
    assert tone in (
        "neutral",
        "reflective",
        "suspenseful",
        "somber",
        "urgent",
        "mysterious",
        "dramatic",
        "informative",
    )


def test_flat_fallback_when_no_signal():
    lines = [_line("l1", "Narrator", "narrative", "The table is by the window.")]
    out, _ = enrich_scene_emotions("s1", lines)
    assert out[0]["emotion"] in ("Flat", "Neutral", "Thoughtful")


def test_character_profiles_aggregate_dialogue_traits():
    lines = [
        _line("l1", "Holmes", "dialogue", "Listen carefully, we must go now."),
        _line("l2", "Watson", "dialogue", "Are you certain we should proceed?"),
    ]
    enriched, _ = enrich_scene_emotions("s1", lines)
    profiles = build_character_performance_profiles([{"scene_id": "s1", "lines": enriched}])
    by_character = {p["character"]: p for p in profiles}
    assert "Holmes" in by_character
    assert "Watson" in by_character
    assert "authority" in by_character["Holmes"]["performance_profile"]
